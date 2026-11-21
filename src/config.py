"""Lê o config.toml (parâmetros do algoritmo) e o .env (portas e modelos, compartilhado com o compose).

Chave de calibração ausente no TOML vira None: o código não inventa valor padrão para limiar ou timeout.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ESTRATEGIAS = ("par", "pergunta", "separado")


class ErroConfig(Exception):
    """Configuração ausente ou inválida."""


@dataclass(frozen=True)
class Prefixos:
    consulta: str = ""
    documento: str = ""


@dataclass(frozen=True)
class Config:
    # .env
    ollama_url: str
    chroma_porta: int
    modelo_geracao: str
    modelo_embedding: str
    # [faq]
    arquivo_faq: Path
    arquivo_categorias: Path
    pares_esperados: int
    # [indice]
    estrategia: str
    arquivo_registros: Path
    # [pedacos]
    pedacos_ativo: bool
    pedaco_tamanho: int | None
    pedaco_sobreposicao: int | None
    # [embedding.prefixos], já resolvido para o modelo do .env
    prefixos: Prefixos
    # [busca]
    k_registros: int
    k_vetores: int
    # [llm]
    temperatura: float
    num_ctx: int
    keep_alive: str | None
    # [confianca] e [limites]: None = a calibrar
    limiar_score: float | None
    limiar_margem: float | None
    mensagem_max_caracteres: int | None
    timeout_etapa_s: float | None
    # só para experimentos (ex.: --sem-prefixos): separa a coleção para não sobrescrever a principal
    sufixo_colecao: str = ""


def ler_env(caminho: Path) -> dict[str, str]:
    """Parser mínimo de .env (CHAVE=valor, # comentário). Evita uma dependência só para isso."""
    if not caminho.exists():
        raise ErroConfig(f"{caminho.name} não encontrado. Copie o .env.example para .env.")
    valores: dict[str, str] = {}
    for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), 1):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        chave, sep, valor = linha.partition("=")
        if not sep:
            raise ErroConfig(f"{caminho.name}, linha {n}: esperado CHAVE=valor")
        valores[chave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def _exigir(dados: dict, secao: str, chave: str):
    try:
        return dados[secao][chave]
    except KeyError:
        raise ErroConfig(f"config.toml: falta [{secao}] {chave}") from None


def _prefixos(dados: dict, modelo: str) -> Prefixos:
    tabela = dados.get("embedding", {}).get("prefixos", {})
    # "nomic-embed-text" vale também para "nomic-embed-text:latest"
    p = tabela.get(modelo) or tabela.get(modelo.split(":")[0]) or {}
    return Prefixos(consulta=p.get("consulta", ""), documento=p.get("documento", ""))


def com_modelo_embedding(cfg: Config, modelo: str, raiz: Path = RAIZ) -> Config:
    """Troca o modelo de embedding e os prefixos que valem para ele."""
    with open(raiz / "config.toml", "rb") as f:
        dados = tomllib.load(f)
    return replace(cfg, modelo_embedding=modelo, prefixos=_prefixos(dados, modelo))


def carregar(raiz: Path = RAIZ) -> Config:
    env = ler_env(raiz / ".env")
    for chave in ("OLLAMA_PORTA", "CHROMA_PORTA", "MODELO_GERACAO", "MODELO_EMBEDDING"):
        if not env.get(chave):
            raise ErroConfig(f".env: falta {chave}")

    with open(raiz / "config.toml", "rb") as f:
        dados = tomllib.load(f)

    estrategia = _exigir(dados, "indice", "estrategia")
    if estrategia not in ESTRATEGIAS:
        raise ErroConfig(f"config.toml: estrategia '{estrategia}' inválida; use {', '.join(ESTRATEGIAS)}")

    pedacos = dados.get("pedacos", {})
    ativo = bool(pedacos.get("ativo", False))
    tamanho, sobreposicao = pedacos.get("tamanho"), pedacos.get("sobreposicao")
    if ativo and (tamanho is None or sobreposicao is None):
        raise ErroConfig("config.toml: [pedacos] ativo exige tamanho e sobreposicao calibrados")

    modelo_emb = env["MODELO_EMBEDDING"]
    prefixos = _prefixos(dados, modelo_emb)

    k_registros = int(_exigir(dados, "busca", "k_registros"))
    if k_registros < 2:
        raise ErroConfig("config.toml: k_registros precisa ser >= 2 (a margem compara o 1º com o 2º)")

    confianca = dados.get("confianca", {})
    limites = dados.get("limites", {})
    return Config(
        ollama_url=f"http://127.0.0.1:{int(env['OLLAMA_PORTA'])}",
        chroma_porta=int(env["CHROMA_PORTA"]),
        modelo_geracao=env["MODELO_GERACAO"],
        modelo_embedding=modelo_emb,
        arquivo_faq=raiz / _exigir(dados, "faq", "arquivo"),
        arquivo_categorias=raiz / _exigir(dados, "faq", "categorias"),
        pares_esperados=int(_exigir(dados, "faq", "pares_esperados")),
        estrategia=estrategia,
        arquivo_registros=raiz / _exigir(dados, "indice", "registros"),
        pedacos_ativo=ativo,
        pedaco_tamanho=tamanho,
        pedaco_sobreposicao=sobreposicao,
        prefixos=prefixos,
        k_registros=k_registros,
        k_vetores=int(dados.get("busca", {}).get("k_vetores", 0)),
        temperatura=float(_exigir(dados, "llm", "temperatura")),
        num_ctx=int(_exigir(dados, "llm", "num_ctx")),
        keep_alive=dados["llm"].get("keep_alive"),
        limiar_score=confianca.get("limiar_score"),
        limiar_margem=confianca.get("limiar_margem"),
        mensagem_max_caracteres=limites.get("mensagem_max_caracteres"),
        timeout_etapa_s=limites.get("timeout_etapa_s"),
    )
