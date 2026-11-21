"""Cliente mínimo da API REST do Ollama, com httpx (sem o cliente oficial: endpoints e campos ficam visíveis).

Endpoints usados: POST /api/embed, POST /api/chat, POST /api/show, GET /api/tags, GET /api/version.
Os campos *_duration da API vêm em nanossegundos; aqui tudo é convertido para milissegundos.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


class ErroOllama(Exception):
    """Ollama fora do ar, timeout ou resposta inesperada."""


def _ms(ns: int | None) -> float | None:
    return None if ns is None else ns / 1e6


@dataclass
class ResultadoEmbed:
    vetores: list[list[float]]
    total_ms: float | None  # medido pelo servidor
    carga_ms: float | None  # tempo para carregar o modelo na memória (~0 se já estava carregado)
    tokens: int | None


@dataclass
class ResultadoChat:
    texto: str
    primeiro_token_ms: float | None  # medido no cliente: do envio até o 1º pedaço de texto
    total_ms: float | None  # servidor
    carga_ms: float | None
    tokens_prompt: int | None
    prompt_ms: float | None
    tokens_saida: int | None
    saida_ms: float | None
    motivo_fim: str | None  # "stop" ou "length" (cortou por limite de tokens)
    extras: dict = field(default_factory=dict)


class Ollama:
    def __init__(self, url: str, timeout_s: float | None = None, keep_alive: str | None = None) -> None:
        # timeout None = sem limite. O timeout por etapa do rascunho ainda é "a medir" (config.toml).
        self._http = httpx.Client(base_url=url, timeout=timeout_s)
        self._keep_alive = keep_alive

    def _post(self, caminho: str, corpo: dict) -> dict:
        try:
            r = self._http.post(caminho, json=corpo)
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException as e:
            raise ErroOllama(f"timeout em {caminho}") from e
        except httpx.HTTPError as e:
            raise ErroOllama(f"{caminho}: {e}") from e

    def versao(self) -> str:
        try:
            return self._http.get("/api/version", timeout=5).json()["version"]
        except httpx.HTTPError as e:
            raise ErroOllama(f"Ollama não responde em {self._http.base_url}: {e}") from e

    def modelos_instalados(self) -> dict[str, str]:
        """nome -> digest."""
        return {m["name"]: m["digest"] for m in self._http.get("/api/tags").json()["models"]}

    def detalhes(self, modelo: str) -> dict:
        return self._post("/api/show", {"model": modelo})

    def embed(self, modelo: str, textos: list[str]) -> ResultadoEmbed:
        corpo: dict = {"model": modelo, "input": textos}
        if self._keep_alive is not None:
            corpo["keep_alive"] = self._keep_alive
        r = self._post("/api/embed", corpo)
        vetores = r.get("embeddings")
        if not vetores or len(vetores) != len(textos):
            raise ErroOllama(f"/api/embed devolveu {len(vetores or [])} vetores para {len(textos)} textos")
        return ResultadoEmbed(vetores, _ms(r.get("total_duration")), _ms(r.get("load_duration")),
                              r.get("prompt_eval_count"))

    def chat(self, modelo: str, mensagens: list[dict], *, formato: dict | None = None,
             opcoes: dict | None = None) -> ResultadoChat:
        """Chat em streaming: permite medir o tempo até o primeiro token, que é o que o cliente sente."""
        corpo: dict = {"model": modelo, "messages": mensagens, "stream": True}
        if formato is not None:
            corpo["format"] = formato  # esquema JSON: o Ollama restringe a saída a ele
        if opcoes:
            corpo["options"] = opcoes
        if self._keep_alive is not None:
            corpo["keep_alive"] = self._keep_alive

        partes: list[str] = []
        primeiro: float | None = None
        final: dict = {}
        inicio = time.perf_counter()
        try:
            with self._http.stream("POST", "/api/chat", json=corpo) as r:
                r.raise_for_status()
                for linha in r.iter_lines():
                    if not linha:
                        continue
                    pedaco = json.loads(linha)
                    if "error" in pedaco:
                        raise ErroOllama(f"/api/chat: {pedaco['error']}")
                    texto = pedaco.get("message", {}).get("content", "")
                    if texto and primeiro is None:
                        primeiro = (time.perf_counter() - inicio) * 1000
                    partes.append(texto)
                    if pedaco.get("done"):
                        final = pedaco
        except httpx.TimeoutException as e:
            raise ErroOllama("timeout em /api/chat") from e
        except httpx.HTTPError as e:
            raise ErroOllama(f"/api/chat: {e}") from e
        if not final:
            raise ErroOllama("/api/chat terminou sem a mensagem final (done)")

        return ResultadoChat(
            texto="".join(partes),
            primeiro_token_ms=primeiro,
            total_ms=_ms(final.get("total_duration")),
            carga_ms=_ms(final.get("load_duration")),
            tokens_prompt=final.get("prompt_eval_count"),
            prompt_ms=_ms(final.get("prompt_eval_duration")),
            tokens_saida=final.get("eval_count"),
            saida_ms=_ms(final.get("eval_duration")),
            motivo_fim=final.get("done_reason"),
        )
