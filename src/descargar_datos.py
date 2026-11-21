"""Descarga los datos crudos a ``data/raw/`` y registra la evidencia de la descarga.

Uso:  python src/descargar_datos.py [--forzar]

Si un archivo ya existe no se vuelve a descargar (a menos que se use --forzar). Al final se escribe
``data/raw/registro_descarga.json`` con URL, fecha, filas, columnas y peso de cada archivo, que el notebook
muestra en la sección de Business Understanding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, FUENTES  # noqa: E402

REGISTRO = DATA_RAW / "registro_descarga.json"

# Problema de infraestructura documentado en el README: el CDN de Transfermarkt (Cloudflare R2) responde
# 403 Forbidden al User-Agent por defecto de urllib ("Python-urllib/3.x"). Se envía un User-Agent descriptivo
# que identifica al proyecto (no se suplanta a ningún navegador).
CABECERAS = {"User-Agent": "Vanegas-BigData-ProyectoFinal/1.0 (proyecto academico; descarga de datos abiertos)"}


def descargar(url: str, destino: Path, reintentos: int = 3) -> None:
    """Descarga ``url`` a ``destino`` con reintentos (escribe primero a .part para no dejar archivos a medias)."""
    temporal = destino.with_suffix(destino.suffix + ".part")
    for intento in range(1, reintentos + 1):
        try:
            peticion = urllib.request.Request(url, headers=CABECERAS)
            with urllib.request.urlopen(peticion, timeout=120) as respuesta, open(temporal, "wb") as salida:
                while bloque := respuesta.read(1 << 20):
                    salida.write(bloque)
            temporal.replace(destino)
            return
        except Exception as error:  # noqa: BLE001 - se reintenta cualquier fallo de red
            print(f"   intento {intento}/{reintentos} falló: {error}")
            time.sleep(3 * intento)
    raise RuntimeError(f"No se pudo descargar {url}")


def perfil_archivo(ruta: Path) -> dict:
    """Filas, columnas y peso en disco/memoria de un CSV (comprimido o no)."""
    tabla = pd.read_csv(ruta, low_memory=False)
    return {
        "filas": int(len(tabla)),
        "columnas": int(tabla.shape[1]),
        "peso_disco_mb": round(ruta.stat().st_size / 1e6, 2),
        "peso_memoria_mb": round(tabla.memory_usage(deep=True).sum() / 1e6, 2),
    }


def main(forzar: bool = False) -> dict:
    registro = json.loads(REGISTRO.read_text(encoding="utf-8")) if REGISTRO.exists() else {}
    for nombre, info in FUENTES.items():
        destino = DATA_RAW / nombre
        if destino.exists() and not forzar:
            print(f"[ok] {nombre} ya existe ({destino.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"[..] descargando {nombre} desde {info['url']}")
            descargar(info["url"], destino)
            registro.setdefault(nombre, {})["fecha_descarga"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entrada = registro.setdefault(nombre, {})
        entrada.setdefault("fecha_descarga", datetime.fromtimestamp(destino.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"))
        entrada.update({"url": info["url"], "fuente": info["fuente"], "pagina": info["pagina"], **perfil_archivo(destino)})
    REGISTRO.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(v["filas"] for v in registro.values())
    print(f"Total de filas crudas: {total:,}")
    return registro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true", help="volver a descargar aunque el archivo exista")
    main(parser.parse_args().forzar)
