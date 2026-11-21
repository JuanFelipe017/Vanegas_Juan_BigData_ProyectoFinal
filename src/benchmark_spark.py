"""Benchmark pandas vs. Spark sobre los MISMOS datos crudos (justificación de arquitectura con datos propios).

Se ejecuta en el equipo del autor (fuera de Docker), porque la imagen de procesamiento NO incluye Spark ni Java:
esa es justamente la decisión que este benchmark justifica. Resultado: reports/benchmark_pandas_spark.json

Uso: python src/benchmark_spark.py   (requiere pyspark y Java 17)
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SALIDA = C.REPORTS / "benchmark_pandas_spark.json"


def cronometrar(funcion, repeticiones: int = 3) -> tuple[float, object]:
    tiempos, resultado = [], None
    for _ in range(repeticiones):
        t = time.perf_counter()
        resultado = funcion()
        tiempos.append(time.perf_counter() - t)
    return round(min(tiempos), 3), resultado


def medir_pandas() -> dict:
    ruta = str(C.DATA_RAW / "Matches.csv")
    t_lectura, m = cronometrar(lambda: pd.read_csv(ruta, low_memory=False))
    t_lectura_valores, v = cronometrar(lambda: pd.read_csv(C.DATA_RAW / "player_valuations.csv.gz"))

    def agregar():
        x = m.assign(anio=m["MatchDate"].str.slice(0, 4))
        return x.groupby(["Division", "anio"]).agg(partidos=("FTResult", "size"), goles=("FTHome", "mean"),
                                                    local=("FTResult", lambda s: (s == "H").mean())).reset_index()
    t_agregar, _ = cronometrar(agregar)

    def unir():
        return v.groupby("current_club_id")["market_value_in_eur"].sum().reset_index().merge(
            v[["player_id", "current_club_id"]].drop_duplicates("player_id"), on="current_club_id")
    t_unir, _ = cronometrar(unir)
    t_parquet, _ = cronometrar(lambda: pd.read_parquet(C.PARQUET_PARTIDOS))
    return {"lectura_csv_partidos_s": t_lectura, "lectura_csv_valoraciones_s": t_lectura_valores,
            "agregacion_liga_anio_s": t_agregar, "agregacion_y_join_valoraciones_s": t_unir,
            "lectura_parquet_limpio_s": t_parquet, "memoria_partidos_mb": round(m.memory_usage(deep=True).sum() / 1e6, 1)}


def medir_spark() -> dict:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    t = time.perf_counter()
    spark = (SparkSession.builder.master("local[*]").appName("benchmark-futbol")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", str(2 * (os.cpu_count() or 2)))
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t_arranque = time.perf_counter() - t
    ruta = str(C.DATA_RAW / "Matches.csv")

    t_lectura, df = cronometrar(lambda: spark.read.csv(ruta, header=True, inferSchema=True).cache())
    df.count()

    def agregar():
        return (df.withColumn("anio", F.substring("MatchDate", 1, 4)).groupBy("Division", "anio")
                .agg(F.count("*").alias("partidos"), F.avg("FTHome").alias("goles"),
                     F.avg(F.when(F.col("FTResult") == "H", 1).otherwise(0)).alias("local")).toPandas())
    t_agregar, _ = cronometrar(agregar)
    v = spark.read.csv(str(C.DATA_RAW / "player_valuations.csv.gz"), header=True, inferSchema=True).cache()
    v.count()

    def unir():
        suma = v.groupBy("current_club_id").agg(F.sum("market_value_in_eur").alias("valor"))
        return suma.join(v.select("player_id", "current_club_id").dropDuplicates(["player_id"]), "current_club_id").count()
    t_unir, _ = cronometrar(unir)
    version = spark.version
    spark.stop()
    return {"version": version, "arranque_sesion_s": round(t_arranque, 3), "lectura_csv_partidos_s": t_lectura,
            "agregacion_liga_anio_s": t_agregar, "agregacion_y_join_valoraciones_s": t_unir,
            "configuracion": {"master": "local[*]", "spark.driver.memory": "4g",
                              "spark.sql.shuffle.partitions": 2 * (os.cpu_count() or 2)}}


def main() -> dict:
    try:
        import psutil  # noqa: F401
        ram = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        ram = None
    resultado = {"maquina": {"sistema": platform.platform(), "cpu_logicos": os.cpu_count(), "ram_gb": ram,
                             "python": platform.python_version(), "pandas": pd.__version__},
                 "fecha": time.strftime("%Y-%m-%d %H:%M"), "pandas": medir_pandas()}
    try:
        resultado["spark"] = medir_spark()
    except Exception as error:  # noqa: BLE001
        resultado["spark"] = {"error": str(error)[:500]}
    SALIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    main()
