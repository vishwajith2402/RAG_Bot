from __future__ import annotations


def get_system_stats() -> dict[str, float | str]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
            "memory_percent": float(memory.percent),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}

