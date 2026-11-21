"""Parser do faq_interno.md e do categorias.toml.

Formato do FAQ (conferido no arquivo): título, uma linha de descrição que contém "Última revisão: MM/AAAA",
depois pares de linhas pergunta/resposta. Linhas vazias são ignoradas. A pergunta é a linha que termina
em "?"; tudo antes da primeira pergunta é cabeçalho.

Política (decidida): falha se a estrutura ou o número de pares estiver errado; data ausente só gera aviso.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

RE_REVISAO = re.compile(r"Última revisão:\s*(\d{2}/\d{4})")


class ErroFAQ(Exception):
    """O FAQ ou o arquivo de categorias não tem a estrutura esperada."""


@dataclass(frozen=True)
class Registro:
    """Registro pai: o par pergunta + resposta inteiro. O id é o número do par (1 = primeira pergunta)."""

    id: int
    pergunta: str
    resposta: str
    categorias: tuple[str, ...] = ()

    @property
    def texto(self) -> str:
        return f"{self.pergunta}\n{self.resposta}"


@dataclass(frozen=True)
class FAQ:
    registros: tuple[Registro, ...]
    revisao: str | None  # "MM/AAAA", ou None se o cabeçalho não trouxer a data
    hash: str  # sha256 do arquivo: vai para o registros.json e para o Chroma, para detectar dessincronia
    especiais: dict[str, str]  # intenções que não vêm do FAQ e escalam direto

    def por_id(self, id_registro: int) -> Registro:
        return self.registros[id_registro - 1]


def ler_faq(arquivo: Path, pares_esperados: int) -> tuple[list[tuple[str, str]], str | None, str]:
    """Devolve (pares, revisao, hash). Sem categorias: isso é com o carregar()."""
    bruto = arquivo.read_bytes()
    linhas = [l.strip() for l in bruto.decode("utf-8-sig").splitlines() if l.strip()]

    inicio = next((i for i, l in enumerate(linhas) if l.endswith("?")), None)
    if inicio is None:
        raise ErroFAQ(f"{arquivo.name}: nenhuma pergunta (linha terminada em '?') encontrada")

    cabecalho = " ".join(linhas[:inicio])
    achado = RE_REVISAO.search(cabecalho)
    revisao = achado.group(1) if achado else None
    if revisao is None:
        warnings.warn(f"{arquivo.name}: 'Última revisão: MM/AAAA' não encontrada no cabeçalho", stacklevel=2)

    corpo = linhas[inicio:]
    if len(corpo) % 2:
        raise ErroFAQ(f"{arquivo.name}: {len(corpo)} linhas depois do cabeçalho; esperado um número par")

    pares = []
    for i in range(0, len(corpo), 2):
        pergunta, resposta = corpo[i], corpo[i + 1]
        n = i // 2 + 1
        if not pergunta.endswith("?"):
            raise ErroFAQ(f"{arquivo.name}: o par {n} não começa com uma pergunta: {pergunta[:60]!r}")
        if resposta.endswith("?"):
            raise ErroFAQ(f"{arquivo.name}: a resposta do par {n} parece outra pergunta: {resposta[:60]!r}")
        pares.append((pergunta, resposta))

    if len(pares) != pares_esperados:
        raise ErroFAQ(f"{arquivo.name}: {len(pares)} pares encontrados; esperado {pares_esperados}")
    return pares, revisao, hashlib.sha256(bruto).hexdigest()


def ler_categorias(arquivo: Path, n_pares: int) -> tuple[dict[int, tuple[str, ...]], dict[str, str]]:
    with open(arquivo, "rb") as f:
        dados = tomllib.load(f)
    brutos = dados.get("registros")
    if not isinstance(brutos, dict):
        raise ErroFAQ(f"{arquivo.name}: falta a tabela [registros]")

    categorias: dict[int, tuple[str, ...]] = {}
    for chave, valor in brutos.items():
        if not chave.isdigit():
            raise ErroFAQ(f"{arquivo.name}: chave '{chave}' não é número de par")
        if isinstance(valor, str):
            valor = [valor]
        if not valor or not all(isinstance(v, str) and v for v in valor):
            raise ErroFAQ(f"{arquivo.name}: par {chave} sem categoria válida")
        categorias[int(chave)] = tuple(valor)

    esperados = set(range(1, n_pares + 1))
    if set(categorias) != esperados:
        faltam = sorted(esperados - set(categorias))
        sobram = sorted(set(categorias) - esperados)
        raise ErroFAQ(f"{arquivo.name}: pares não batem com o FAQ (faltam {faltam}, sobram {sobram})")

    especiais = dict(dados.get("especiais", {}))
    colisao = {c for cs in categorias.values() for c in cs} & set(especiais)
    if colisao:
        raise ErroFAQ(f"{arquivo.name}: {sorted(colisao)} é categoria de registro e intenção especial ao mesmo tempo")
    return categorias, especiais


def carregar(arquivo_faq: Path, arquivo_categorias: Path, pares_esperados: int) -> FAQ:
    pares, revisao, hash_faq = ler_faq(arquivo_faq, pares_esperados)
    categorias, especiais = ler_categorias(arquivo_categorias, len(pares))
    registros = tuple(
        Registro(id=n, pergunta=p, resposta=r, categorias=categorias[n]) for n, (p, r) in enumerate(pares, 1)
    )
    return FAQ(registros=registros, revisao=revisao, hash=hash_faq, especiais=especiais)


def dividir_em_pedacos(texto: str, tamanho: int, sobreposicao: int) -> list[str]:
    """Divide em pedaços de `tamanho` palavras, repetindo `sobreposicao` palavras entre vizinhos.

    Desligada no config ([pedacos] ativo = false): no FAQ atual cada par cabe num pedaço só.
    Unidade em palavras, não tokens: os tokens dependem do tokenizador do modelo de embedding.
    """
    if tamanho <= 0 or not 0 <= sobreposicao < tamanho:
        raise ValueError("exige tamanho > 0 e 0 <= sobreposicao < tamanho")
    palavras = texto.split()
    if len(palavras) <= tamanho:
        return [" ".join(palavras)]
    passo = tamanho - sobreposicao
    pedacos = []
    for inicio in range(0, len(palavras), passo):
        pedacos.append(" ".join(palavras[inicio : inicio + tamanho]))
        if inicio + tamanho >= len(palavras):
            break
    return pedacos


if __name__ == "__main__":
    # conferência rápida: python -m src.faq
    from src import config

    cfg = config.carregar()
    faq = carregar(cfg.arquivo_faq, cfg.arquivo_categorias, cfg.pares_esperados)
    print(f"{len(faq.registros)} registros | revisão {faq.revisao} | sha256 {faq.hash[:12]}…")
    for r in faq.registros:
        print(f"{r.id:>2}  {'+'.join(r.categorias):<38} {r.pergunta}")
    print("especiais:", ", ".join(faq.especiais))
