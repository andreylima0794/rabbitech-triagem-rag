"""Cronômetro por etapa e estatísticas de latência (mín/mediana/média/p95/máx)."""

from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager


class Cronometro:
    """Acumula o tempo de parede de cada etapa, em milissegundos.

    Uso: with crono.etapa("embedding"): ...
    Se a mesma etapa rodar duas vezes (ex.: segunda busca depois do esclarecimento), use nomes diferentes.
    """

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def etapa(self, nome: str):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            self.ms[nome] = self.ms.get(nome, 0.0) + (time.perf_counter() - inicio) * 1000

    @property
    def total_ms(self) -> float:
        return sum(self.ms.values())


def percentil(valores: list[float], p: float) -> float:
    """Percentil pelo método nearest-rank (sem interpolação): sempre um valor que de fato ocorreu."""
    if not valores:
        return math.nan
    ordenados = sorted(valores)
    posicao = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicao - 1]


def resumo(valores: list[float]) -> dict[str, float]:
    if not valores:
        return {"n": 0, "min": math.nan, "mediana": math.nan, "media": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "n": len(valores),
        "min": min(valores),
        "mediana": statistics.median(valores),
        "media": statistics.fmean(valores),
        "p95": percentil(valores, 95),
        "max": max(valores),
    }
