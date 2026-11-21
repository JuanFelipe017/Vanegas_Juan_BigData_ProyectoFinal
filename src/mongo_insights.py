"""Documento de análisis en MongoDB (estructura de la Actividad 1, ampliada con el modelo).

Todas las cifras de los insights se calculan aquí a partir del Parquet limpio, los cubos OLAP y las métricas del
modelo: nada está escrito a mano. Se guarda una copia JSON en ``reports/analisis_mongodb.json`` como evidencia.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

DOC_ID = "analisis_futbol_apuestas"
JSON_COPIA = C.REPORTS / "analisis_mongodb.json"


def num(x: float, dec: int = 1) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==================================================================================================
# Conexión
# ==================================================================================================
def conectar(reintentos: int = 20, espera: float = 3.0):
    """Devuelve (cliente, colección). Reintenta porque en Docker Mongo puede tardar unos segundos en aceptar conexiones."""
    from pymongo import MongoClient
    ultimo_error = None
    for _ in range(reintentos):
        try:
            cliente = MongoClient(C.MONGO_URI, serverSelectionTimeoutMS=3000)
            cliente.admin.command("ping")
            return cliente, cliente[C.MONGO_DB][C.MONGO_COLECCION]
        except Exception as error:  # noqa: BLE001
            ultimo_error = error
            time.sleep(espera)
    raise ConnectionError(f"No se pudo conectar a MongoDB en {C.MONGO_HOST}:{C.MONGO_PORT}: {ultimo_error}")


# ==================================================================================================
# Construcción del documento
# ==================================================================================================
def _limpiar(obj):
    """Convierte tipos de numpy/pandas a tipos nativos (BSON no acepta numpy)."""
    if isinstance(obj, dict):
        return {str(k): _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def estadisticas_descriptivas(m: pd.DataFrame) -> dict:
    variables = ["total_goles", "goles_local", "goles_visitante", "cuota_local", "cuota_empate", "cuota_visitante",
                 "margen_casa", "diferencia_elo", "valor_plantilla_local", "tiros_local", "amarillas_local"]
    desc = m[variables].astype("float64").describe().T.round(3)
    return {
        "variables": desc.reset_index().rename(columns={"index": "variable"}).to_dict(orient="records"),
        "distribucion_resultados_pct": (m["resultado"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "partidos_por_continente": m["continente"].value_counts().to_dict(),
        "partidos_por_nivel": m["nivel"].value_counts().sort_index().rename(lambda x: f"nivel_{x}").to_dict(),
    }


def agregaciones(cubo_lt: pd.DataFrame, cubo_pd: pd.DataFrame, cubo_eq: pd.DataFrame) -> list[dict]:
    grandes = cubo_lt[cubo_lt["liga_codigo"].isin(["E0", "SP1", "I1", "D1", "F1"])]
    ultima = grandes[grandes["temporada_inicio"] == 2025]
    local_decada = cubo_pd[(cubo_pd["resultado"] == "Local") & (cubo_pd["nivel"] == 1)]
    pivote = local_decada.pivot_table(index="pais", columns="decada", values="pct_dentro_grupo").round(1)
    campeones = cubo_eq[(cubo_eq["posicion"] == 1) & (cubo_eq["nivel"] == 1) & (cubo_eq["temporada_inicio"] == 2025)]
    return [
        {"nombre": "Liga × temporada", "tipo_olap": "slice (temporada 2025/26) sobre el cubo liga × temporada",
         "dimensiones": ["continente", "pais", "liga", "nivel", "temporada"],
         "medidas": ["partidos", "pct_local", "pct_empate", "pct_visitante", "goles_prom", "margen_casa_prom", "acierto_favorito"],
         "filas_cubo": int(len(cubo_lt)),
         "muestra": ultima[["liga", "partidos", "pct_local", "pct_empate", "pct_visitante", "goles_prom", "margen_casa_prom", "acierto_favorito"]].to_dict(orient="records")},
        {"nombre": "País × década × nivel × resultado", "tipo_olap": "dice (nivel 1, resultado Local) y pivote país × década",
         "dimensiones": ["continente", "pais", "decada", "nivel", "resultado"], "medidas": ["partidos", "goles_prom", "pct_dentro_grupo"],
         "filas_cubo": int(len(cubo_pd)),
         "muestra": pivote.reset_index().to_dict(orient="records")},
        {"nombre": "Equipo × temporada", "tipo_olap": "drill-down liga → equipo (líderes 2025/26)",
         "dimensiones": ["pais", "liga", "temporada", "equipo"], "medidas": ["partidos", "puntos", "gf", "gc", "ppg", "valor_plantilla_meur"],
         "filas_cubo": int(len(cubo_eq)),
         "muestra": campeones[["liga", "equipo", "partidos", "puntos", "gf", "gc", "ppg", "valor_plantilla_meur"]].to_dict(orient="records")},
    ]


def construir_insights(m: pd.DataFrame, met: dict) -> tuple[list[dict], list[dict]]:
    grandes = m[m["liga_codigo"].isin(["E0", "SP1", "I1", "D1", "F1"]) & (m["temporada_inicio"] <= 2025)]
    pct_local = grandes.groupby("temporada_inicio")["resultado"].apply(lambda s: (s == "Local").mean() * 100)
    inicio, fin, covid = pct_local.loc[2000:2002].mean(), pct_local.loc[2023:2025].mean(), pct_local.loc[2020]

    europa = m[(m["nivel"] == 1) & (m["continente"] == "Europa")].groupby("temporada_inicio")["margen_casa"].mean() * 100
    lb = met["lineas_base"]
    casa = lb["Favorito de la casa"]["prueba"]
    ingenua = lb["Siempre local"]["prueba"]
    mejor_con = met["resultados"]["con_cuotas"]["_mejor"]
    mejor_sin = met["resultados"]["sin_cuotas"]["_mejor"]
    r_con = met["resultados"]["con_cuotas"][mejor_con["modelo"]]["prueba"]
    r_sin = met["resultados"]["sin_cuotas"][mejor_sin["modelo"]]["prueba"]
    imp_sin = met["importancias"]["sin_cuotas"]
    sim_con = met["simulacion"]["con_cuotas"]["umbral_5pct"]
    sim_casa = met["simulacion"].get("casa", {}).get("umbral_5pct", {})
    sim_sin = met["simulacion"]["sin_cuotas"]["umbral_5pct"]

    d = m.dropna(subset=["log_ratio_valor"])
    deciles = pd.qcut(d["log_ratio_valor"], 10, labels=False)
    local_dec = d.groupby(deciles)["resultado"].apply(lambda s: (s == "Local").mean() * 100)
    primera = m[m["nivel"] == 1].groupby("liga", observed=True)["total_goles"].mean().sort_values()
    empates = (m["resultado"] == "Empate").mean() * 100

    insights = [
        {"id": 1, "tema": "Ventaja local", "titulo": "La ventaja de jugar en casa se está encogiendo",
         "texto": f"En las 5 grandes ligas el local ganaba el {num(inicio, 1)} % de los partidos a inicios de los 2000 y hoy gana el "
                  f"{num(fin, 1)} %. En 2020/21, con estadios vacíos por la COVID-19, cayó a {num(covid, 1)} %: el público explica "
                  f"buena parte de la ventaja.", "valor": round(fin - inicio, 2), "unidad": "puntos porcentuales"},
        {"id": 2, "tema": "Mercado de apuestas", "titulo": "Las casas están calibradas y cobran cada vez menos margen",
         "texto": f"Cuando Bet365 da 60 % a un resultado, ocurre cerca del 60 % de las veces (diagrama de calibración). Su margen en "
                  f"primeras europeas bajó de {num(europa.iloc[0], 1)} % (2000) a {num(europa.loc[:2025].iloc[-1], 1)} % (2025): el "
                  f"mercado es más eficiente y competitivo.", "valor": round(float(europa.loc[:2025].iloc[-1]), 3), "unidad": "% de margen"},
        {"id": 3, "tema": "Modelo", "titulo": "El modelo con cuotas iguala a la casa; sin cuotas queda cerca",
         "texto": f"En {num(met['particion']['prueba'], 0)} partidos de prueba (2023/24 en adelante), {mejor_con['modelo']} con cuotas logra "
                  f"log-loss {num(r_con['log_loss'], 4)} frente a {num(casa['log_loss'], 4)} de la casa y {num(ingenua['log_loss'], 4)} de "
                  f"la línea base ingenua. El modelo de fútbol puro ({mejor_sin['modelo']}) acierta {num(r_sin['accuracy'] * 100, 1)} % vs. "
                  f"{num(casa['accuracy'] * 100, 1)} % de la casa: solo con datos públicos se llega a {num(abs(r_sin['accuracy'] - casa['accuracy']) * 100, 1)} pp del mercado.",
         "valor": round(r_con["log_loss"] - casa["log_loss"], 5), "unidad": "diferencia de log-loss"},
        {"id": 4, "tema": "Modelo", "titulo": "La diferencia de Elo es la variable reina del fútbol puro",
         "texto": f"En el modelo sin cuotas, '{imp_sin[0]['etiqueta']}' es por lejos la variable más importante (importancia por "
                  f"permutación {num(imp_sin[0]['importancia'], 4)}), seguida de '{imp_sin[1]['etiqueta']}' y '{imp_sin[2]['etiqueta']}'. "
                  f"La forma reciente aporta poco una vez se conoce el Elo.", "valor": round(imp_sin[0]["importancia"], 5), "unidad": "Δ log-loss"},
        {"id": 5, "tema": "Valor de mercado", "titulo": "La plata sí juega: el valor de plantilla separa resultados",
         "texto": f"Cuando el local tiene la plantilla relativamente más cara (decil superior) gana el {num(local_dec.iloc[-1], 1)} % de "
                  f"las veces; cuando tiene la más barata, solo el {num(local_dec.iloc[0], 1)} %. El valor de Transfermarkt enriquece "
                  f"el Elo con información de calidad individual.", "valor": round(float(local_dec.iloc[-1] - local_dec.iloc[0]), 2), "unidad": "puntos porcentuales"},
        {"id": 6, "tema": "Empates", "titulo": "El empate es el resultado más difícil de predecir",
         "texto": f"El {num(empates, 1)} % de los partidos termina empatado, pero casi nunca es el resultado más probable. Con el umbral de "
                  f"decisión ajustado (predecir empate si p ≥ {num(mejor_con['umbral_empate'], 3)}) el recall de empates sube a "
                  f"{num(mejor_con['prueba_con_umbral']['por_clase']['Empate']['recall'] * 100, 1)} % y el F1 macro pasa de "
                  f"{num(r_con['f1_macro'], 3)} a {num(mejor_con['prueba_con_umbral']['f1_macro'], 3)}.",
         "valor": round(mejor_con["prueba_con_umbral"]["f1_macro"], 4), "unidad": "F1 macro"},
        {"id": 7, "tema": "Apuestas", "titulo": "Ganarle a la casa es posible solo en el margen y con la mejor cuota",
         "texto": f"Apostando solo cuando el valor esperado supera 5 % y con la mejor cuota del mercado, el modelo con cuotas hizo "
                  f"{num(sim_con['apuestas'], 0)} apuestas con ROI {num(sim_con['roi'] * 100, 2)} %. La estrategia de control (probabilidades "
                  f"de la propia casa) obtuvo ROI {num(sim_casa.get('roi', 0) * 100, 2)} % y el modelo sin cuotas "
                  f"{num(sim_sin['roi'] * 100, 2)} %. "
                  + ("Como el control pierde y el modelo gana, el modelo aporta información más allá de buscar la mejor cuota, "
                     "aunque el margen es pequeño y volátil." if sim_con["roi"] > 0 >= sim_casa.get("roi", 0)
                     else "La ventaja depende sobre todo de buscar la mejor cuota, no del modelo."),
         "valor": round(sim_con["roi"], 5), "unidad": "ROI"},
        {"id": 8, "tema": "Ligas", "titulo": "Hay ligas de goles y ligas de candado",
         "texto": f"Entre primeras divisiones, {primera.index[-1]} promedia {num(primera.iloc[-1], 2)} goles por partido y "
                  f"{primera.index[0]} solo {num(primera.iloc[0], 2)}. Un modelo de goles debe tener en cuenta la liga.",
         "valor": round(float(primera.iloc[-1] - primera.iloc[0]), 3), "unidad": "goles por partido"},
    ]
    recomendaciones = [
        {"id": 1, "dirigida_a": "Apostadores", "texto": "No apostar con las cuotas de una sola casa: sin comparar cuotas el ROI esperado es "
         f"negativo (margen ≈ {num(europa.loc[:2025].iloc[-1], 1)} %). Si se apuesta, hacerlo solo con valor esperado positivo, la mejor "
         "cuota disponible y montos fijos pequeños, porque las rachas negativas son largas."},
        {"id": 2, "dirigida_a": "Clubes y analistas", "texto": "Usar el Elo (propio o de ClubElo) y el valor de plantilla como base de "
         "cualquier pronóstico de puntos o de presupuesto deportivo; la forma de los últimos 5 partidos aporta poco y suele sobrevalorarse."},
        {"id": 3, "dirigida_a": "Equipo de datos", "texto": "Re-entrenar el modelo cada temporada: la ventaja local cambió (COVID, VAR) y un "
         "modelo entrenado con los 2000 sobreestima al local. Vigilar la calibración con el diagrama de la pestaña Modelo."},
        {"id": 4, "dirigida_a": "Equipo de datos", "texto": "Para mejorar el empate, probar un modelo específico de goles (Poisson "
         "bivariado o Dixon-Coles) y variables de estilo (tiros, xG) cuando la fuente las tenga completas."},
    ]
    return insights, recomendaciones


def construir_documento(m: pd.DataFrame | None = None, met: dict | None = None, etl: dict | None = None,
                        visualizaciones: list[dict] | None = None) -> dict:
    m = pd.read_parquet(C.PARQUET_PARTIDOS) if m is None else m
    met = json.loads(C.JSON_MODELO.read_text(encoding="utf-8")) if met is None else met
    etl = json.loads(C.JSON_ETL.read_text(encoding="utf-8")) if etl is None else etl
    registro = json.loads((C.DATA_RAW / "registro_descarga.json").read_text(encoding="utf-8"))
    cubo_lt, cubo_pd, cubo_eq = (pd.read_parquet(p) for p in (C.CUBO_LIGA_TEMPORADA, C.CUBO_PAIS_DECADA, C.CUBO_EQUIPO_TEMPORADA))
    insights, recomendaciones = construir_insights(m, met)
    mejor = met["resultados"]["con_cuotas"]["_mejor"]
    mejor_sin = met["resultados"]["sin_cuotas"]["_mejor"]

    def resumen_metricas(r):
        return {k: r[k] for k in ["accuracy", "balanced_accuracy", "f1_macro", "log_loss", "brier", "roc_auc_ovr"]}

    doc = {
        "_id": DOC_ID,
        "metadata": {
            "fecha_analisis": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "autores": [C.AUTOR],
            "curso": "Gerencia y almacenamiento en Big Data — Fundación Universitaria Los Libertadores",
            "proyecto": "Fútbol y apuestas: ¿se le puede ganar a la casa? (38 ligas, 2000–2026)",
            "video": C.VIDEO_URL,
            "dataset": {
                "fuente": "Football-Data.co.uk + ClubElo + Transfermarkt",
                "url": [v["url"] for v in registro.values()],
                "registros": etl["filas_crudas_total"], "columnas": sum(p["columnas"] for p in etl["perfil_crudo"].values()),
                "archivos": registro,
                "partidos_limpios": etl["partidos_limpios"], "columnas_limpias": etl["columnas_limpias"],
                "periodo": f"{etl['fecha_min']} a {etl['fecha_max']}", "ligas": etl["ligas"], "equipos": etl["equipos"],
            },
            "herramienta": "pandas (justificación en el notebook, sección 2)",
        },
        "calidad_datos": {
            "problemas_detectados": [p for p in etl["diagnostico"] + etl["diagnostico_transfermarkt"] if p["registros"] > 0],
            "registros_eliminados": int(etl["perfil_crudo"]["partidos"]["filas"] - etl["partidos_limpios"]),
            "pasos_etl": etl["pasos"],
        },
        "estadisticas_descriptivas": estadisticas_descriptivas(m),
        "agregaciones": agregaciones(cubo_lt, cubo_pd, cubo_eq),
        "modelo": {
            "tarea": met["tarea"], "algoritmo": mejor["modelo"],
            "algoritmo_sin_cuotas": mejor_sin["modelo"],
            "particion": met["particion"],
            "metricas": resumen_metricas(met["resultados"]["con_cuotas"][mejor["modelo"]]["prueba"]),
            "metricas_con_umbral_empate": {**resumen_metricas(mejor["prueba_con_umbral"]), "umbral_empate": mejor["umbral_empate"]},
            "metricas_sin_cuotas": resumen_metricas(met["resultados"]["sin_cuotas"][mejor_sin["modelo"]]["prueba"]),
            "linea_base": {k: resumen_metricas(v["prueba"]) for k, v in met["lineas_base"].items()},
            "comparacion": {conj: {mod: resumen_metricas(r["prueba"]) for mod, r in res.items() if not mod.startswith("_")}
                            for conj, res in met["resultados"].items()},
            "variables_importantes": met["importancias"]["con_cuotas"][:10],
            "variables_importantes_sin_cuotas": met["importancias"]["sin_cuotas"][:10],
            "simulacion_apuestas": {k: v["umbral_5pct"] for k, v in met["simulacion"].items()},
            "criterios_exito": met["criterios_exito"],
        },
        "insights": insights,
        "visualizaciones_generadas": visualizaciones or [],
        "recomendaciones": recomendaciones,
    }
    return _limpiar(doc)


def guardar(doc: dict) -> str:
    """Inserta/reemplaza el documento en MongoDB y guarda la copia JSON."""
    cliente, col = conectar()
    try:
        col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        col.create_index("insights.tema")
    finally:
        cliente.close()
    JSON_COPIA.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc["_id"]


def leer() -> dict:
    cliente, col = conectar()
    try:
        return col.find_one({"_id": DOC_ID})
    finally:
        cliente.close()


def consultas_ejemplo(col) -> dict:
    """Las 3+ consultas que se muestran en el notebook."""
    return {
        "1_metricas_modelo": col.find_one({"_id": DOC_ID}, {"_id": 0, "modelo.algoritmo": 1, "modelo.metricas": 1, "modelo.linea_base": 1}),
        "2_insights_modelo_o_apuestas": list(col.aggregate([
            {"$match": {"_id": DOC_ID}}, {"$unwind": "$insights"},
            {"$match": {"insights.tema": {"$in": ["Modelo", "Apuestas"]}}},
            {"$project": {"_id": 0, "tema": "$insights.tema", "titulo": "$insights.titulo"}}])),
        "3_top5_variables": list(col.aggregate([
            {"$match": {"_id": DOC_ID}}, {"$unwind": "$modelo.variables_importantes"},
            {"$sort": {"modelo.variables_importantes.importancia": -1}}, {"$limit": 5},
            {"$project": {"_id": 0, "variable": "$modelo.variables_importantes.etiqueta",
                          "importancia": "$modelo.variables_importantes.importancia"}}])),
        "4_problemas_calidad_mas_frecuentes": list(col.aggregate([
            {"$match": {"_id": DOC_ID}}, {"$unwind": "$calidad_datos.problemas_detectados"},
            {"$sort": {"calidad_datos.problemas_detectados.registros": -1}}, {"$limit": 5},
            {"$project": {"_id": 0, "problema": "$calidad_datos.problemas_detectados.problema",
                          "registros": "$calidad_datos.problemas_detectados.registros"}}])),
        "5_conteo_insights_por_tema": list(col.aggregate([
            {"$match": {"_id": DOC_ID}}, {"$unwind": "$insights"},
            {"$group": {"_id": "$insights.tema", "cantidad": {"$sum": 1}}}, {"$sort": {"cantidad": -1}}])),
    }


if __name__ == "__main__":
    documento = construir_documento()
    print("Guardado en MongoDB:", guardar(documento))
