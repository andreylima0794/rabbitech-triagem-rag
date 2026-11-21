"""Hardware e versões usados numa medição. Vai para resultados/ e para o repositório público.

De propósito, NÃO coleta hostname, nome de usuário nem caminhos: só o que explica o número de latência.
Sem psutil: cada SO tem uma fonte nativa para CPU e RAM.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from importlib.metadata import version


def _cpu() -> str:
    sistema = platform.system()
    try:
        if sistema == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sistema == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "desconhecida"


def _ram_gb() -> float | None:
    sistema = platform.system()
    try:
        if sistema == "Windows":
            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            s = Status()
            s.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.ullTotalPhys / 2**30, 1)
        if sistema == "Darwin":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30, 1)
        with open("/proc/meminfo", encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("MemTotal"):
                    return round(int(linha.split()[1]) / 2**20, 1)
    except Exception:
        pass
    return None


def _gpu() -> list[dict]:
    try:
        saida = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return []  # sem NVIDIA (ex.: Mac): o Ollama roda em CPU ou Metal, registrado pelo usuário no README
    gpus = []
    for linha in saida.strip().splitlines():
        nome, memoria, driver = [c.strip() for c in linha.split(",")]
        gpus.append({"nome": nome, "vram_gb": round(int(memoria) / 1024, 1), "driver": driver})
    return gpus


def coletar(ollama, cfg, chroma_versao: str | None = None) -> dict:
    instalados = ollama.modelos_instalados()

    def modelo(nome: str) -> dict:
        chave = nome if nome in instalados else f"{nome}:latest"
        d = ollama.detalhes(nome)
        info = d.get("model_info", {})
        contexto_max = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        return {
            "nome": nome,
            "digest": instalados.get(chave, "")[:12],
            "parametros": d.get("details", {}).get("parameter_size"),
            "quantizacao": d.get("details", {}).get("quantization_level"),
            "contexto_maximo_do_modelo": contexto_max,
        }

    return {
        "so": f"{platform.system()} {platform.release()}",
        "cpu": _cpu(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "python": sys.version.split()[0],
        "ollama": ollama.versao(),
        "chroma_versao_api": chroma_versao,
        "chromadb_cliente": version("chromadb"),
        "modelo_geracao": modelo(cfg.modelo_geracao),
        "modelo_embedding": modelo(cfg.modelo_embedding),
    }
