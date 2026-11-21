"""Linha de comando.

  python -m src.cli index   [--estrategia par|pergunta|separado]
  python -m src.cli ask     "mensagem do cliente"
  python -m src.cli chat
  python -m src.cli bench   [--perguntas perguntas_exemplo.csv] [--repeticoes 3] [--conjunto dev] [--forcar-geracao]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import ambiente, armazenamento, config, faq as faq_mod, tempos
from src.ollama_api import ErroOllama, Ollama
from src.pipeline import ESCALAR, ESCLARECER, Pipeline, Resultado

console = Console()
ETAPAS = ("regras", "classificacao", "embedding", "busca", "decisao", "geracao", "verificacao")


# --- montagem ---------------------------------------------------------------------------------------

def _preparar(args) -> tuple[config.Config, faq_mod.FAQ, Ollama, object]:
    cfg = config.carregar()
    if getattr(args, "estrategia", None):
        cfg = replace(cfg, estrategia=args.estrategia)
    if getattr(args, "modelo_embedding", None):
        cfg = config.com_modelo_embedding(cfg, args.modelo_embedding)
    if getattr(args, "sem_prefixos", False):
        cfg = replace(cfg, prefixos=config.Prefixos(), sufixo_colecao="__sem-prefixos")
    faq = faq_mod.carregar(cfg.arquivo_faq, cfg.arquivo_categorias, cfg.pares_esperados)
    ollama = Ollama(cfg.ollama_url, cfg.timeout_etapa_s, cfg.keep_alive)
    ollama.versao()  # falha cedo, com mensagem clara, se o Ollama não estiver de pé
    cliente = armazenamento.conectar(cfg.chroma_porta)
    try:
        cliente.heartbeat()
    except Exception as e:
        raise ErroOllama(f"Chroma não responde na porta {cfg.chroma_porta}: {e}") from None
    return cfg, faq, ollama, cliente


def _pipeline(args) -> tuple[Pipeline, config.Config, Ollama, object]:
    cfg, faq, ollama, cliente = _preparar(args)
    nome = armazenamento.nome_colecao(cfg.modelo_embedding, cfg.estrategia) + cfg.sufixo_colecao
    colecao = armazenamento.abrir_colecao(cliente, nome, faq.hash)
    registros = armazenamento.ler_registros(cfg.arquivo_registros, faq.hash)
    return Pipeline(cfg, faq, ollama, colecao, registros), cfg, ollama, cliente


# --- index ------------------------------------------------------------------------------------------

def documentos(faq: faq_mod.FAQ, cfg: config.Config) -> list[tuple[str, str, dict]]:
    """(id do vetor, texto, metadados) conforme a estratégia. Os filhos sempre apontam para o id do pai."""
    docs = []
    for r in faq.registros:
        partes = {
            "par": [("par", r.texto)],
            "pergunta": [("pergunta", r.pergunta)],
            "separado": [("pergunta", r.pergunta), ("resposta", r.resposta)],
        }[cfg.estrategia]
        for tipo, texto in partes:
            pedacos = (faq_mod.dividir_em_pedacos(texto, cfg.pedaco_tamanho, cfg.pedaco_sobreposicao)
                       if cfg.pedacos_ativo else [texto])
            for i, pedaco in enumerate(pedacos):
                docs.append((f"{r.id}-{tipo}-{i}", pedaco, {"id_registro": r.id, "tipo": tipo, "pedaco": i}))
    return docs


def cmd_index(args) -> None:
    cfg, faq, ollama, cliente = _preparar(args)
    docs = documentos(faq, cfg)
    inicio = time.perf_counter()
    emb = ollama.embed(cfg.modelo_embedding, [cfg.prefixos.documento + texto for _, texto, _ in docs])
    nome = armazenamento.nome_colecao(cfg.modelo_embedding, cfg.estrategia) + cfg.sufixo_colecao
    colecao = armazenamento.recriar_colecao(cliente, nome, faq.hash, cfg.modelo_embedding, cfg.estrategia)
    colecao.add(ids=[d[0] for d in docs], embeddings=emb.vetores, metadatas=[d[2] for d in docs])
    armazenamento.salvar_registros(cfg.arquivo_registros, faq)
    total = (time.perf_counter() - inicio) * 1000
    console.print(
        f"[green]ok[/] {len(faq.registros)} registros, {len(docs)} vetores ({len(emb.vetores[0])} dimensões) "
        f"na coleção [bold]{nome}[/] | FAQ rev. {faq.revisao} | {emb.tokens} tokens de embedding | {total:.0f} ms"
    )


# --- ask / chat --------------------------------------------------------------------------------------

def _mostrar(res: Resultado, pipe: Pipeline, detalhes: bool = True) -> None:
    cor = {"RESPONDER": "green", "ESCLARECER": "yellow", "ESCALAR": "red"}[res.decisao]
    console.print(f"[bold {cor}]{res.decisao}[/] ({res.motivo})")
    console.print(res.texto)
    if not detalhes:
        return
    if res.intencao:
        console.print(f"[dim]intenção: {res.intencao}[/]")
    for c in res.candidatos:
        console.print(f"[dim]  #{c.id_registro:<2} {c.score:.4f} ({c.tipo})  {pipe.registros[c.id_registro].pergunta}[/]")
    if res.sinais and res.sinais.margem is not None:
        console.print(f"[dim]margem {res.sinais.margem:.4f} | coerente {res.sinais.coerente}[/]")
    if res.geracao and res.decisao == ESCALAR:
        console.print(f"[dim]geração (não enviada; verificação: {res.verificacao or 'ok'}): {res.geracao.texto}[/]")
    console.print("[dim]" + " | ".join(f"{k} {v:.0f} ms" for k, v in res.tempos_ms.items()) + "[/]")
    if res.decisao == ESCALAR:
        console.print(f"[dim]resumo para o atendente: {json.dumps(res.resumo_para_humano, ensure_ascii=False)}[/]")


def cmd_ask(args) -> None:
    pipe, *_ = _pipeline(args)
    _mostrar(pipe.atender(args.mensagem, forcar_geracao=args.forcar_geracao), pipe)


def cmd_chat(args) -> None:
    """Uma conversa: no máximo uma rodada de esclarecimento, depois escala."""
    pipe, *_ = _pipeline(args)
    console.print("[dim]Mensagem vazia sai.[/]")
    while True:
        msg = console.input("[bold]cliente> [/]").strip()
        if not msg:
            return
        res = pipe.atender(msg)
        _mostrar(res, pipe, detalhes=args.detalhes)
        if res.decisao == ESCLARECER:
            resposta = console.input("[bold]cliente> [/]").strip()
            # volta ao início com a mensagem original + a resposta, e esclarecimentos = 1
            _mostrar(pipe.atender(f"{msg} {resposta}", esclarecimentos=1), pipe, detalhes=args.detalhes)


# --- bench -------------------------------------------------------------------------------------------

def _linhas_perguntas(caminho: Path, conjunto: str | None) -> list[dict]:
    with open(caminho, encoding="utf-8", newline="") as f:
        linhas = [l for l in csv.DictReader(f) if not l["id"].startswith("#")]
    return [l for l in linhas if conjunto in (None, l["conjunto"])]


def _acertou(res: Resultado, linha: dict) -> bool:
    """O top-1 é o registro esperado (para as linhas que têm um)."""
    esperado = linha["registro_esperado"].strip()
    return bool(esperado) and bool(res.candidatos) and res.candidatos[0].id_registro == int(esperado)


def cmd_bench(args) -> None:
    pipe, cfg, ollama, cliente = _pipeline(args)
    linhas = _linhas_perguntas(Path(args.perguntas), args.conjunto)
    if not linhas:
        sys.exit(f"nenhuma pergunta em {args.perguntas} para o conjunto {args.conjunto}")

    # aquecimento: a 1ª chamada carrega os modelos na GPU (dezenas de segundos). Medido e registrado à parte.
    inicio = time.perf_counter()
    aquec = pipe.atender(linhas[0]["mensagem"], forcar_geracao=True)
    frio_ms = (time.perf_counter() - inicio) * 1000

    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
    pasta = Path("resultados")
    pasta.mkdir(exist_ok=True)
    rotulo_prefixos = "com-prefixos" if cfg.prefixos.consulta or cfg.prefixos.documento else "sem-prefixos"
    prefixo = f"{carimbo}_{cfg.modelo_embedding.replace(':', '-')}_{rotulo_prefixos}_{cfg.estrategia}"
    amb = ambiente.coletar(ollama, cfg, chroma_versao=cliente.get_version())
    env = config.ler_env(config.RAIZ / ".env")  # a imagem diz a versão real do servidor; a API diz "1.0.0"
    amb["imagens_docker"] = {"ollama": env.get("OLLAMA_IMAGEM"), "chroma": env.get("CHROMA_IMAGEM")}
    amb["execucao"] = {
        "estrategia": cfg.estrategia, "perguntas": Path(args.perguntas).name, "conjunto": args.conjunto,
        "repeticoes": args.repeticoes, "forcar_geracao": args.forcar_geracao,
        "limiar_score": cfg.limiar_score, "limiar_margem": cfg.limiar_margem, "k_registros": cfg.k_registros,
        "num_ctx": cfg.num_ctx, "temperatura": cfg.temperatura,
        "aquecimento_ms": round(frio_ms), "aquecimento_tempos_ms": {k: round(v) for k, v in aquec.tempos_ms.items()},
    }
    (pasta / f"{prefixo}_ambiente.json").write_text(json.dumps(amb, ensure_ascii=False, indent=2), encoding="utf-8")

    campos = ["rep", "id", "grupo", "conjunto", "esperado", "registro_esperado", "decisao", "motivo", "intencao",
              "top1", "top1_tipo", "score1", "top2", "score2", "margem", "coerente", "top1_correto",
              "verificacao", "tokens_prompt_classif", "tokens_prompt_geracao", "tokens_saida_geracao",
              "primeiro_token_geracao_ms", "carga_ms", "busca2_ms", "decisao2", "motivo2"] + \
             [f"{e}_ms" for e in ETAPAS] + ["total_ms"]
    arquivo_csv = pasta / f"{prefixo}_bench.csv"
    tempos_por_etapa: dict[str, list[float]] = {e: [] for e in (*ETAPAS, "total", "primeiro_token")}
    resultados: list[tuple[dict, Resultado]] = []

    with open(arquivo_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        with console.status("") as status:
            for rep in range(1, args.repeticoes + 1):
                for i, linha in enumerate(linhas, 1):
                    status.update(f"rep {rep}/{args.repeticoes} · {i}/{len(linhas)} · {linha['id']}")
                    res = pipe.atender(linha["mensagem"], forcar_geracao=args.forcar_geracao)
                    resultados.append((linha, res))
                    c = res.candidatos
                    reg = {
                        "rep": rep, "id": linha["id"], "grupo": linha["grupo"], "conjunto": linha["conjunto"],
                        "esperado": linha["esperado"], "registro_esperado": linha["registro_esperado"],
                        "decisao": res.decisao, "motivo": res.motivo, "intencao": res.intencao,
                        "top1": c[0].id_registro if c else "", "top1_tipo": c[0].tipo if c else "",
                        "score1": f"{c[0].score:.4f}" if c else "",
                        "top2": c[1].id_registro if len(c) > 1 else "",
                        "score2": f"{c[1].score:.4f}" if len(c) > 1 else "",
                        "margem": f"{res.sinais.margem:.4f}" if res.sinais and res.sinais.margem is not None else "",
                        "coerente": res.sinais.coerente if res.sinais else "",
                        "top1_correto": _acertou(res, linha) if linha["registro_esperado"].strip() else "",
                        "verificacao": res.verificacao or ("ok" if res.geracao else ""),
                        "tokens_prompt_classif": res.classificacao.tokens_prompt if res.classificacao else "",
                        "tokens_prompt_geracao": res.geracao.tokens_prompt if res.geracao else "",
                        "tokens_saida_geracao": res.geracao.tokens_saida if res.geracao else "",
                        "primeiro_token_geracao_ms": f"{res.geracao.primeiro_token_ms:.1f}"
                        if res.geracao and res.geracao.primeiro_token_ms else "",
                        "carga_ms": f"{(res.classificacao.carga_ms or 0):.1f}" if res.classificacao else "",
                    }
                    # ESCLARECER: mede a segunda passada com a resposta do cliente, se o CSV trouxer uma
                    if res.decisao == ESCLARECER and linha.get("resposta_esclarecimento", "").strip():
                        r2 = pipe.atender(f"{linha['mensagem']} {linha['resposta_esclarecimento']}", esclarecimentos=1)
                        reg.update(busca2_ms=f"{sum(r2.tempos_ms.values()):.1f}", decisao2=r2.decisao, motivo2=r2.motivo)
                    for e in ETAPAS:
                        if e in res.tempos_ms:
                            reg[f"{e}_ms"] = f"{res.tempos_ms[e]:.1f}"
                            tempos_por_etapa[e].append(res.tempos_ms[e])
                    reg["total_ms"] = f"{sum(res.tempos_ms.values()):.1f}"
                    tempos_por_etapa["total"].append(sum(res.tempos_ms.values()))
                    if res.geracao and res.geracao.primeiro_token_ms:
                        tempos_por_etapa["primeiro_token"].append(res.geracao.primeiro_token_ms)
                    w.writerow(reg)

    _imprimir_bench(tempos_por_etapa, resultados, frio_ms, args)
    console.print(f"\n[dim]gravado: {arquivo_csv} e {prefixo}_ambiente.json[/]")


def _imprimir_bench(tempos_etapa, resultados, frio_ms, args) -> None:
    t = Table(title=f"Latência por etapa (ms) · {args.repeticoes} repetição(ões) · aquecimento {frio_ms:.0f} ms fora")
    for col in ("etapa", "n", "mín", "mediana", "média", "p95", "máx"):
        t.add_column(col, justify="right" if col != "etapa" else "left")
    for etapa, valores in tempos_etapa.items():
        if valores:
            r = tempos.resumo(valores)
            t.add_row(etapa, str(r["n"]), *(f"{r[k]:.1f}" for k in ("min", "mediana", "media", "p95", "max")))
    console.print(t)

    decisoes: dict[str, int] = {}
    for _, res in resultados:
        decisoes[res.decisao] = decisoes.get(res.decisao, 0) + 1
    console.print("decisões: " + ", ".join(f"{k} {v}" for k, v in sorted(decisoes.items())))

    # Dados para calibrar (rascunho 7.5): score e margem de quem acertou × de quem não devia responder.
    com_alvo = [(l, r) for l, r in resultados if l["registro_esperado"].strip() and r.candidatos]
    if com_alvo:
        acertos = sum(_acertou(r, l) for l, r in com_alvo)
        console.print(f"top-1 correto: {acertos}/{len(com_alvo)} ({100 * acertos / len(com_alvo):.0f}%)")
    s = Table(title="Score do 1º e margem, por grupo (insumo para o limiar; nada é escolhido aqui)")
    for col in ("grupo", "n", "score mín", "score mediana", "score máx", "margem mín", "margem mediana"):
        s.add_column(col, justify="right" if col != "grupo" else "left")
    grupos: dict[str, list] = {}
    for l, r in resultados:
        if r.sinais and r.sinais.score is not None:
            chave = l["grupo"] + (" (top-1 errado)" if l["registro_esperado"].strip() and not _acertou(r, l) else "")
            grupos.setdefault(chave, []).append(r.sinais)
    for g, sinais in sorted(grupos.items()):
        sc = tempos.resumo([x.score for x in sinais])
        mg = tempos.resumo([x.margem for x in sinais])
        s.add_row(g, str(sc["n"]), f"{sc['min']:.4f}", f"{sc['mediana']:.4f}", f"{sc['max']:.4f}",
                  f"{mg['min']:.4f}", f"{mg['mediana']:.4f}")
    console.print(s)


# --- calibrar ----------------------------------------------------------------------------------------

GRADE_SCORE = [round(0.40 + 0.05 * i, 2) for i in range(11)]  # 0,40 a 0,90: só a grade da tabela, não um limiar
GRADE_MARGEM = [round(0.01 * i, 2) for i in range(11)]  # 0,00 a 0,10


def cmd_calibrar(args) -> None:
    """Só a busca (sem LLM), em TODAS as mensagens dos grupos dentro/fora/ambiguo do conjunto escolhido.

    Mostra, para cada limiar candidato, quanto de cada grupo passaria. Não escolhe nada: a escolha é da autora.
    Lembrete: no pipeline real, o classificador barra parte do fora de escopo antes da busca.
    """
    cfg, faq, ollama, cliente = _preparar(args)
    nome = armazenamento.nome_colecao(cfg.modelo_embedding, cfg.estrategia) + cfg.sufixo_colecao
    colecao = armazenamento.abrir_colecao(cliente, nome, faq.hash)
    linhas = [l for l in _linhas_perguntas(Path(args.perguntas), args.conjunto)
              if l["grupo"] in ("dentro", "fora", "ambiguo")]
    emb = ollama.embed(cfg.modelo_embedding, [cfg.prefixos.consulta + l["mensagem"] for l in linhas])

    casos = []
    for linha, vetor in zip(linhas, emb.vetores):
        c = armazenamento.agrupar_por_registro(armazenamento.buscar(colecao, vetor, cfg.k_vetores))
        esperado = linha["registro_esperado"].strip()
        casos.append({"id": linha["id"], "grupo": linha["grupo"], "top1": c[0].id_registro, "score": c[0].score,
                      "margem": c[0].score - (c[1].score if len(c) > 1 else 0.0),
                      "correto": (c[0].id_registro == int(esperado)) if esperado else None})

    por_grupo = {g: [x for x in casos if x["grupo"] == g] for g in ("dentro", "fora", "ambiguo")}
    certos = [x for x in por_grupo["dentro"] if x["correto"]]
    errados = [x for x in por_grupo["dentro"] if not x["correto"]]
    prefixo = "com" if cfg.prefixos.consulta or cfg.prefixos.documento else "sem"
    console.print(f"[bold]{cfg.modelo_embedding} / {cfg.estrategia}[/] (prefixos: {prefixo}) · conjunto {args.conjunto} · "
                  f"dentro n={len(por_grupo['dentro'])} (top-1 certo {len(certos)}, errado {len(errados)}) · "
                  f"fora n={len(por_grupo['fora'])} · ambíguo n={len(por_grupo['ambiguo'])}")

    def pct(parte, todo):
        return f"{len(parte)}/{len(todo)} ({100 * len(parte) / len(todo):.0f}%)" if todo else "-"

    t = Table(title="Por limiar de SCORE do 1º: quantos passariam (score >= limiar)")
    for col in ("limiar", "dentro, top-1 certo", "dentro, top-1 ERRADO", "fora de escopo", "ambíguo"):
        t.add_column(col, justify="right")
    for lim in GRADE_SCORE:
        t.add_row(f"{lim:.2f}", *(pct([x for x in g if x["score"] >= lim], g)
                                   for g in (certos, errados, por_grupo["fora"], por_grupo["ambiguo"])))
    console.print(t)

    t = Table(title="Por limiar de MARGEM (1º - 2º): quantos passariam (margem >= limiar)")
    for col in ("limiar", "dentro, top-1 certo", "dentro, top-1 ERRADO", "fora de escopo", "ambíguo"):
        t.add_column(col, justify="right")
    for lim in GRADE_MARGEM:
        t.add_row(f"{lim:.2f}", *(pct([x for x in g if x["margem"] >= lim], g)
                                   for g in (certos, errados, por_grupo["fora"], por_grupo["ambiguo"])))
    console.print(t)

    arquivo = Path("resultados") / (f"{datetime.now():%Y%m%d-%H%M%S}_calibracao_"
                                    f"{cfg.modelo_embedding.replace(':', '-')}_{cfg.estrategia}_{prefixo}-prefixos.csv")
    with open(arquivo, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(casos[0]))
        w.writeheader()
        w.writerows({**x, "score": f"{x['score']:.4f}", "margem": f"{x['margem']:.4f}"} for x in casos)
    console.print(f"[dim]gravado: {arquivo}[/]")


# --- main --------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m src.cli", description="Triagem de suporte RabbITech (Ollama + RAG)")
    sub = p.add_subparsers(dest="comando", required=True)

    def com_estrategia(sp):
        sp.add_argument("--estrategia", choices=config.ESTRATEGIAS, help="sobrepõe o config.toml")
        sp.add_argument("--modelo-embedding", help="sobrepõe o MODELO_EMBEDDING do .env (comparação D2)")
        sp.add_argument("--sem-prefixos", action="store_true",
                        help="desliga os prefixos de tarefa (usa uma coleção própria, '__sem-prefixos')")
        return sp

    com_estrategia(sub.add_parser("index", help="indexa o FAQ (recria a coleção e o registros.json)"))
    sp = com_estrategia(sub.add_parser("ask", help="uma mensagem, mostra decisão, candidatos e tempos"))
    sp.add_argument("mensagem")
    sp.add_argument("--forcar-geracao", action="store_true", help="gera mesmo sem limiar calibrado (não envia)")
    sp = com_estrategia(sub.add_parser("chat", help="conversa com uma rodada de esclarecimento"))
    sp.add_argument("--detalhes", action="store_true")
    sp = com_estrategia(sub.add_parser("bench", help="mede latência e coleta score/margem para calibrar"))
    sp.add_argument("--perguntas", default="perguntas_exemplo.csv")
    sp.add_argument("--conjunto", choices=("dev", "teste"), default="dev",
                    help="o conjunto 'teste' é para rodar uma vez, no fim (rascunho, seção 10)")
    sp.add_argument("--repeticoes", type=int, default=1)
    sp.add_argument("--forcar-geracao", action="store_true", help="mede a geração mesmo quando a decisão escala")

    sp = com_estrategia(sub.add_parser("calibrar", help="tabelas por limiar de score e margem (só a busca)"))
    sp.add_argument("--perguntas", default="perguntas_exemplo.csv")
    sp.add_argument("--conjunto", choices=("dev", "teste"), default="dev")

    args = p.parse_args(argv)
    try:
        {"index": cmd_index, "ask": cmd_ask, "chat": cmd_chat, "bench": cmd_bench,
         "calibrar": cmd_calibrar}[args.comando](args)
    except (config.ErroConfig, faq_mod.ErroFAQ, armazenamento.ErroIndice, ErroOllama) as e:
        console.print(f"[red]erro:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
