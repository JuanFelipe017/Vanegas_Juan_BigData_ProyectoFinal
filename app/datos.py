"""Carga de datos YA procesados (Parquet, modelos, MongoDB). El dashboard nunca repite el ETL ni re-entrena."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402

COLUMNAS_PARTIDOS = [
    "fecha", "temporada", "temporada_inicio", "decada", "mes", "liga_codigo", "liga", "pais", "continente", "nivel",
    "local", "visitante", "goles_local", "goles_visitante", "total_goles", "resultado", "resultado_ht", "mas_de_2_5", "ambos_marcan",
    "cuota_local", "cuota_empate", "cuota_visitante", "cuota_max_local", "cuota_max_empate", "cuota_max_visitante",
    "prob_casa_local", "prob_casa_empate", "prob_casa_visitante", "margen_casa", "favorito_casa", "acierto_favorito", "sorpresa",
    "elo_clubelo_local", "elo_clubelo_visitante", "elo_propio_local", "elo_propio_visitante", "diferencia_elo", "elo_imputado",
    "valor_plantilla_local", "valor_plantilla_visitante", "log_ratio_valor", "puntos_prom5_local", "puntos_prom5_visitante",
    "tiros_local", "tiros_visitante", "tiros_arco_local", "tiros_arco_visitante", "corners_local", "corners_visitante",
    "amarillas_local", "amarillas_visitante", "rojas_local", "rojas_visitante", "faltas_local", "faltas_visitante",
    "post_paron", "dif_viajeros", "dif_descanso_real", "dif_carga_min14",
]


def listo() -> bool:
    return C.PARQUET_PARTIDOS.exists() and C.JSON_MODELO.exists() and C.MODELO_CON_CUOTAS.exists()


@st.cache_data(show_spinner="Cargando partidos…")
def partidos() -> pd.DataFrame:
    m = pd.read_parquet(C.PARQUET_PARTIDOS, columns=COLUMNAS_PARTIDOS)
    for c in ["local", "visitante", "liga", "pais", "continente", "liga_codigo", "temporada", "decada"]:
        m[c] = m[c].astype(str).astype("category")
    return m


@st.cache_data
def cubos() -> dict[str, pd.DataFrame]:
    return {"liga_temporada": pd.read_parquet(C.CUBO_LIGA_TEMPORADA), "pais_decada": pd.read_parquet(C.CUBO_PAIS_DECADA),
            "equipo_temporada": pd.read_parquet(C.CUBO_EQUIPO_TEMPORADA)}


@st.cache_data
def predicciones() -> pd.DataFrame:
    return pd.read_parquet(C.PARQUET_PREDICCIONES)


@st.cache_data
def estado_equipos() -> pd.DataFrame:
    return pd.read_parquet(C.PARQUET_ESTADO)


@st.cache_data
def elo_clubelo() -> pd.DataFrame:
    return pd.read_parquet(C.PARQUET_ELO)


@st.cache_data
def equipos_partido() -> pd.DataFrame:
    return pd.read_parquet(C.PARQUET_EQUIPOS, columns=["fecha", "equipo", "liga_codigo", "temporada", "elo_propio", "puntos", "gf", "gc", "es_local", "rival"])


@st.cache_data
def metricas() -> dict:
    return json.loads(C.JSON_MODELO.read_text(encoding="utf-8"))


@st.cache_data
def resumen_etl() -> dict:
    return json.loads(C.JSON_ETL.read_text(encoding="utf-8"))


@st.cache_resource
def modelos() -> dict:
    return {"con_cuotas": joblib.load(C.MODELO_CON_CUOTAS), "sin_cuotas": joblib.load(C.MODELO_SIN_CUOTAS)}


@st.cache_data(ttl=120, show_spinner="Consultando MongoDB…")
def documento_mongo() -> tuple[dict | None, str]:
    """Documento de análisis desde MongoDB; si Mongo no responde, usa la copia JSON y lo avisa."""
    try:
        from pymongo import MongoClient
        cliente = MongoClient(C.MONGO_URI, serverSelectionTimeoutMS=2500)
        doc = cliente[C.MONGO_DB][C.MONGO_COLECCION].find_one({"_id": "analisis_futbol_apuestas"})
        cliente.close()
        if doc:
            return doc, "MongoDB"
    except Exception:  # noqa: BLE001
        pass
    copia = C.REPORTS / "analisis_mongodb.json"
    if copia.exists():
        return json.loads(copia.read_text(encoding="utf-8")), "copia JSON (MongoDB no disponible)"
    return None, "sin datos"


# ==================================================================================================
# Filtros globales (panel flotante)
# ==================================================================================================
FILTROS_DEFECTO = {"f_temporadas": (2000, 2026), "f_continentes": [], "f_paises": [], "f_ligas": [], "f_niveles": [],
                   "f_animaciones": True}


def iniciar_filtros() -> None:
    for k, v in FILTROS_DEFECTO.items():
        st.session_state.setdefault(k, v)


def filtrar(m: pd.DataFrame) -> pd.DataFrame:
    s = st.session_state
    a, b = s["f_temporadas"]
    mascara = m["temporada_inicio"].between(a, b)
    if s["f_continentes"]:
        mascara &= m["continente"].isin(s["f_continentes"])
    if s["f_paises"]:
        mascara &= m["pais"].isin(s["f_paises"])
    if s["f_ligas"]:
        mascara &= m["liga"].isin(s["f_ligas"])
    if s["f_niveles"]:
        mascara &= m["nivel"].isin(s["f_niveles"])
    return m[mascara]


def descripcion_filtros() -> str:
    s = st.session_state
    partes = [f"{s['f_temporadas'][0]}–{s['f_temporadas'][1]}"]
    for k, nombre in [("f_continentes", "continentes"), ("f_paises", "países"), ("f_ligas", "ligas"), ("f_niveles", "niveles")]:
        if s[k]:
            partes.append(f"{len(s[k])} {nombre}" if len(s[k]) > 2 else ", ".join(map(str, s[k])))
    return " · ".join(partes)


# ==================================================================================================
# Predicción de un partido nuevo con el modelo guardado
# ==================================================================================================
def fila_prediccion(local: pd.Series, visitante: pd.Series, liga_codigo: str, nivel: int, mes: int, descanso_l: float,
                    descanso_v: float, cuotas: tuple[float, float, float] | None,
                    paron: dict | None = None) -> pd.DataFrame:
    """Construye las variables del partido con el mismo significado que en el entrenamiento."""
    vl, vv = local["valor_plantilla"], visitante["valor_plantilla"]
    fila = {
        "diferencia_elo": local["elo_propio"] - visitante["elo_propio"],
        "elo_propio_local": local["elo_propio"], "elo_propio_visitante": visitante["elo_propio"],
        "puntos_prom5_local": local["puntos_prom5"], "puntos_prom5_visitante": visitante["puntos_prom5"],
        "dif_puntos_prom5": local["puntos_prom5"] - visitante["puntos_prom5"],
        "gf_prom5_local": local["gf_prom5"], "gc_prom5_local": local["gc_prom5"],
        "gf_prom5_visitante": visitante["gf_prom5"], "gc_prom5_visitante": visitante["gc_prom5"],
        "ppg_temporada_local": local["ppg_temporada"], "ppg_temporada_visitante": visitante["ppg_temporada"],
        "dif_ppg_temporada": local["ppg_temporada"] - visitante["ppg_temporada"],
        "partidos_temporada_local": local["partidos_temporada"],
        "record_condicion_local": local["record_local"], "record_condicion_visitante": visitante["record_visitante"],
        "dias_descanso_local": descanso_l, "dias_descanso_visitante": descanso_v,
        "log_ratio_valor": np.log1p(vl) - np.log1p(vv) if pd.notna(vl) and pd.notna(vv) else np.nan,
        "valor_plantilla_local": vl, "valor_plantilla_visitante": vv,
        "nivel": nivel, "mes": mes, "liga_codigo": liga_codigo,
        "prob_casa_local": np.nan, "prob_casa_empate": np.nan, "prob_casa_visitante": np.nan, "logit_casa": np.nan,
        "margen_casa": np.nan, "prob_casa_mas25": np.nan, "handicap": np.nan,
        # fatiga: el descanso del formulario es descanso real; la carga de minutos se deja al modelo (NaN = desconocida)
        "descanso_real_local": descanso_l, "descanso_real_visitante": descanso_v, "dif_descanso_real": descanso_l - descanso_v,
        "carga_min14_local": np.nan, "carga_min14_visitante": np.nan, "dif_carga_min14": np.nan,
        "post_paron": 0.0, "convocados_local": 0.0, "convocados_visitante": 0.0,
        "viajeros_local": 0.0, "viajeros_visitante": 0.0, "dif_viajeros": 0.0,
    }
    if paron:
        fila.update({"post_paron": 1.0, **{k: float(v) for k, v in paron.items()},
                     "dif_viajeros": float(paron["viajeros_local"] - paron["viajeros_visitante"])})
    if cuotas:
        inv = np.array([1 / q for q in cuotas])
        p = inv / inv.sum()
        fila.update({"prob_casa_local": p[0], "prob_casa_empate": p[1], "prob_casa_visitante": p[2],
                     "logit_casa": np.log(p[0] / p[2]), "margen_casa": inv.sum() - 1})
    return pd.DataFrame([fila])


def predecir(paquete: dict, fila: pd.DataFrame) -> dict[str, float]:
    columnas = paquete["numericas"] + paquete["categoricas"]
    modelo = paquete["pipeline"]
    p = modelo.predict_proba(fila[columnas])[0]
    return {c: float(p[list(modelo.classes_).index(c)]) for c in paquete["clases"]}
