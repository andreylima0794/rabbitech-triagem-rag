"""Índice: Chroma guarda só os vetores (filhos); o texto completo (pai) fica no registros.json.

Cada vetor leva nos metadados o id do registro pai e o tipo (par, pergunta ou resposta). O código, e não o
LLM, usa esse id para buscar o texto completo. A coleção e o registros.json guardam o hash do FAQ:
se o FAQ mudar sem reindexar, a busca recusa rodar.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.faq import FAQ, Registro


class ErroIndice(Exception):
    """Índice ausente ou fora de sincronia com o FAQ."""


@dataclass(frozen=True)
class Vetor:
    id_registro: int
    tipo: str  # par | pergunta | resposta
    score: float  # 1 - distância cosseno


@dataclass(frozen=True)
class Candidato:
    """Melhor vetor de um registro, depois de agrupar."""

    id_registro: int
    tipo: str
    score: float


def conectar(porta: int) -> chromadb.HttpClient:
    # anonymized_telemetry=False: desde a 1.5.4 o servidor não coleta telemetria de produto, mas o
    # cliente desliga explicitamente mesmo assim.
    return chromadb.HttpClient(host="127.0.0.1", port=porta, settings=Settings(anonymized_telemetry=False))


def nome_colecao(modelo_embedding: str, estrategia: str) -> str:
    """Uma coleção por modelo + estratégia, para as comparações conviverem sem apagar índices."""
    modelo = re.sub(r"[^a-zA-Z0-9._-]", "-", modelo_embedding.removesuffix(":latest"))
    return f"faq__{modelo}__{estrategia}"


def recriar_colecao(cliente, nome: str, hash_faq: str, modelo: str, estrategia: str):
    try:
        cliente.delete_collection(nome)
    except Exception:  # não existia
        pass
    # embedding_function=None: sem isso o Chroma usa a função padrão, que baixa um modelo ONNX da internet.
    return cliente.create_collection(
        nome,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
        metadata={"hash_faq": hash_faq, "modelo_embedding": modelo, "estrategia": estrategia},
    )


def abrir_colecao(cliente, nome: str, hash_faq: str):
    try:
        colecao = cliente.get_collection(nome, embedding_function=None)
    except Exception:
        raise ErroIndice(f"coleção '{nome}' não existe: rode o index") from None
    if colecao.metadata.get("hash_faq") != hash_faq:
        raise ErroIndice(f"coleção '{nome}' foi gerada de outra versão do FAQ: rode o index de novo")
    return colecao


def buscar(colecao, vetor: list[float], n_vetores: int) -> list[Vetor]:
    """n_vetores = 0 busca todos (com 18 registros é barato)."""
    total = colecao.count()
    n = total if n_vetores <= 0 else min(n_vetores, total)
    r = colecao.query(query_embeddings=[vetor], n_results=n, include=["distances", "metadatas"])
    return [
        Vetor(id_registro=int(m["id_registro"]), tipo=m["tipo"], score=1.0 - d)
        for m, d in zip(r["metadatas"][0], r["distances"][0])
    ]


def agrupar_por_registro(vetores: list[Vetor]) -> list[Candidato]:
    """Fica com o melhor vetor de cada registro, para que o 1º e o 2º da margem sejam registros diferentes."""
    melhores: dict[int, Vetor] = {}
    for v in vetores:
        if v.id_registro not in melhores or v.score > melhores[v.id_registro].score:
            melhores[v.id_registro] = v
    return sorted((Candidato(v.id_registro, v.tipo, v.score) for v in melhores.values()),
                  key=lambda c: c.score, reverse=True)


# --- registro pai (registros.json) ---

def salvar_registros(caminho: Path, faq: FAQ) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "hash_faq": faq.hash,
        "revisao": faq.revisao,
        "registros": [
            {"id": r.id, "pergunta": r.pergunta, "resposta": r.resposta, "categorias": list(r.categorias)}
            for r in faq.registros
        ],
    }
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def ler_registros(caminho: Path, hash_faq: str) -> dict[int, Registro]:
    if not caminho.exists():
        raise ErroIndice(f"{caminho.name} não existe: rode o index")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if dados.get("hash_faq") != hash_faq:
        raise ErroIndice(f"{caminho.name} foi gerado de outra versão do FAQ: rode o index de novo")
    return {
        r["id"]: Registro(r["id"], r["pergunta"], r["resposta"], tuple(r["categorias"])) for r in dados["registros"]
    }
