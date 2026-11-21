"""Orquestador del pipeline completo (lo ejecuta el servicio `procesamiento` de Docker al arrancar).

Orden: descarga → ETL → modelo → gráficas → MongoDB → PDF → bandera de listo.

Es idempotente: si el Parquet y los modelos ya existen (por ejemplo, porque vienen en la entrega) no repite el ETL ni
el entrenamiento, pero SIEMPRE vuelve a cargar el documento en MongoDB (el volumen de Mongo puede estar vacío en un
equipo nuevo) y regenera el PDF desde Mongo. Use ``--forzar`` para recalcular todo desde los datos crudos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def paso(titulo: str) -> float:
    print(f"\n{'=' * 90}\n▶ {titulo}\n{'=' * 90}", flush=True)
    return time.perf_counter()


def main(forzar: bool = False) -> None:
    C.BANDERA_OK.unlink(missing_ok=True)
    inicio = time.perf_counter()

    t = paso("1/6 Descarga de datos crudos → data/raw")
    import descargar_datos
    descargar_datos.main(forzar=False)
    print(f"   listo en {time.perf_counter() - t:.1f} s")

    t = paso("2/6 ETL (limpieza, normalización, enriquecimiento, agregación) → data/processed")
    if forzar or not C.PARQUET_PARTIDOS.exists() or not C.JSON_ETL.exists():
        from etl import ejecutar_etl
        ejecutar_etl(verbose=False)
    else:
        print("   Parquet ya existe: se reutiliza (use --forzar para recalcular)")
    print(f"   listo en {time.perf_counter() - t:.1f} s")

    t = paso("3/6 Modelado y evaluación → models/")
    if forzar or not (C.MODELO_CON_CUOTAS.exists() and C.MODELO_SIN_CUOTAS.exists() and C.JSON_MODELO.exists()):
        from modelo import ejecutar_modelado
        ejecutar_modelado()
    else:
        print("   Modelos ya entrenados: se reutilizan (el dashboard nunca re-entrena)")
    print(f"   listo en {time.perf_counter() - t:.1f} s")

    t = paso("4/6 Gráficas estáticas → reports/figuras")
    from graficas import generar_todas
    visualizaciones = generar_todas()
    print(f"   {len(visualizaciones)} figuras en {time.perf_counter() - t:.1f} s")

    t = paso("5/6 Documento de análisis → MongoDB")
    import mongo_insights
    documento = mongo_insights.construir_documento(visualizaciones=visualizaciones)
    mongo_insights.guardar(documento)
    print(f"   documento '{documento['_id']}' guardado en {C.MONGO_DB}.{C.MONGO_COLECCION} "
          f"({len(documento['insights'])} insights, {len(documento['recomendaciones'])} recomendaciones)")

    t = paso("6/6 Reporte PDF (leído desde MongoDB) → reports/")
    import reporte_pdf
    ruta = reporte_pdf.generar(mongo_insights.leer())
    print(f"   {ruta.name} ({ruta.stat().st_size / 1e6:.2f} MB) en {time.perf_counter() - t:.1f} s")

    C.BANDERA_OK.write_text(json.dumps({"segundos": round(time.perf_counter() - inicio, 1)}), encoding="utf-8")
    print(f"\n✔ Pipeline completo en {time.perf_counter() - inicio:.1f} s. Dashboard: http://localhost:8501")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true", help="recalcular ETL y modelos aunque existan")
    main(parser.parse_args().forzar)
