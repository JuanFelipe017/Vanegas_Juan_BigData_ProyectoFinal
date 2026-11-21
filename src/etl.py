"""ETL del proyecto (CRISP-DM: Data Understanding + Data Preparation) con pandas.

Las cuatro técnicas se aplican en este orden y cada paso deja un registro con filas/valores antes y después:

1. **Limpieza**: espacios en nombres, duplicados, cuotas imposibles, márgenes de casa absurdos, estadísticas
   imposibles (más tiros al arco que tiros, 145 faltas, 9 expulsados) y columnas con fuga de información.
2. **Normalización**: nombres de columnas en español, tipos de datos optimizados, fechas, temporada unificada
   (ligas europeas jul–jun y ligas de calendario anual), cuotas → probabilidades sin margen, y nombres de equipos
   homologados entre fuentes (Football-Data ↔ Transfermarkt).
3. **Agregación**: cubos OLAP liga × temporada, país × década × nivel × resultado, y equipo × temporada.
4. **Enriquecimiento**: Elo propio (imputa el 37 % de Elo faltante), forma reciente y descanso calculados sin
   fuga de información, valor de mercado de la plantilla (Transfermarkt) y dimensiones geográficas.

Uso: ``python src/etl.py``  (o ``from etl import ejecutar_etl`` desde el notebook).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# ==================================================================================================
# Registro de pasos (evidencia antes/después para el notebook, MongoDB y el PDF)
# ==================================================================================================
@dataclass
class RegistroETL:
    pasos: list[dict] = field(default_factory=list)

    def anotar(self, tecnica: str, paso: str, antes: int, despues: int, afectados: int, detalle: str) -> None:
        self.pasos.append({
            "tecnica": tecnica, "paso": paso, "antes": int(antes), "despues": int(despues),
            "afectados": int(afectados), "detalle": detalle,
        })

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.pasos)


# ==================================================================================================
# Carga
# ==================================================================================================
def cargar_crudos() -> dict[str, pd.DataFrame]:
    """Lee los seis archivos crudos de ``data/raw``."""
    faltantes = [n for n in C.FUENTES if not (C.DATA_RAW / n).exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan archivos crudos {faltantes}. Ejecute primero: python src/descargar_datos.py")
    return {
        "partidos": pd.read_csv(C.DATA_RAW / "Matches.csv", low_memory=False),
        "elo": pd.read_csv(C.DATA_RAW / "EloRatings.csv"),
        "tm_partidos": pd.read_csv(C.DATA_RAW / "games.csv.gz", low_memory=False),
        "tm_clubes": pd.read_csv(C.DATA_RAW / "clubs.csv.gz"),
        "tm_competiciones": pd.read_csv(C.DATA_RAW / "competitions.csv.gz"),
        "tm_valores": pd.read_csv(C.DATA_RAW / "player_valuations.csv.gz"),
        "tm_apariciones": pd.read_csv(C.DATA_RAW / "appearances.csv.gz", low_memory=False,
                                      usecols=["player_id", "player_club_id", "date", "minutes_played"]),
        "tm_jugadores": pd.read_csv(C.DATA_RAW / "players.csv.gz", low_memory=False,
                                    usecols=["player_id", "country_of_citizenship", "international_caps"]),
        "selecciones": pd.read_csv(C.DATA_RAW / "selecciones_results.csv"),
    }


def memoria_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1e6, 2)


# ==================================================================================================
# Data Understanding: diagnóstico de calidad (antes de limpiar)
# ==================================================================================================
def diagnostico_calidad(m: pd.DataFrame) -> pd.DataFrame:
    """Tabla de problemas de calidad detectados en Matches.csv (conteos sobre los datos crudos)."""
    sobre = 1 / m["OddHome"] + 1 / m["OddDraw"] + 1 / m["OddAway"]
    ratio_max = pd.concat([
        m["MaxHome"] / m["OddHome"], m["MaxDraw"] / m["OddDraw"], m["MaxAway"] / m["OddAway"]], axis=1).max(axis=1)
    problemas = [
        ("Espacios sobrantes en nombres de equipo", (m["HomeTeam"] != m["HomeTeam"].str.strip()) | (m["AwayTeam"] != m["AwayTeam"].str.strip()),
         "Formato", "Recortar con str.strip()"),
        ("Cuotas 1X2 menores a 1,01 (imposibles: pagarían menos de lo apostado)",
         (m[["OddHome", "OddDraw", "OddAway"]] < 1.01).any(axis=1), "Validez", "Anular las 3 cuotas del partido"),
        ("Margen de la casa fuera de [0 %, 30 %] (suma de 1/cuota < 1 o > 1,3)",
         (sobre < 1) | (sobre > 1.3), "Consistencia", "Anular las 3 cuotas del partido"),
        ("Cuota máxima > 3 veces la de Bet365 o < 1,01 (error de captura)",
         (ratio_max > 3) | (m[["MaxHome", "MaxDraw", "MaxAway"]] < 1.01).any(axis=1), "Validez", "Anular cuotas máximas"),
        ("Cuotas máximas de goles > 20 (Over/Under 2,5)", (m["MaxOver25"] > 20) | (m["MaxUnder25"] > 20), "Validez", "Anular"),
        ("Hándicap asiático fuera de ±10 goles (p. ej. −99,9)", m["HandiSize"].abs() > 10, "Validez", "Anular hándicap y sus cuotas"),
        ("Cuotas de hándicap fuera de [1,01; 20]",
         ~m["HandiHome"].between(1.01, 20) & m["HandiHome"].notna() | ~m["HandiAway"].between(1.01, 20) & m["HandiAway"].notna(),
         "Validez", "Anular hándicap y sus cuotas"),
        ("Más tiros al arco que tiros totales", (m["HomeTarget"] > m["HomeShots"]) | (m["AwayTarget"] > m["AwayShots"]),
         "Consistencia", "Anular tiros y tiros al arco del partido"),
        ("Más de 60 faltas de un equipo (máx. real ≈ 35)", (m["HomeFouls"] > 60) | (m["AwayFouls"] > 60), "Atípico", "Anular faltas"),
        ("Más de 5 expulsados de un equipo (el partido se suspende con 6 o menos jugadores)",
         (m["HomeRed"] > 5) | (m["AwayRed"] > 5), "Validez", "Anular tarjetas rojas"),
        ("Elo de ClubElo faltante", m["HomeElo"].isna() | m["AwayElo"].isna(), "Completitud",
         "Imputar con Elo propio calculado (enriquecimiento)"),
        ("Estadísticas de partido faltantes (tiros, faltas, córners, tarjetas)", m["HomeShots"].isna(), "Completitud",
         "Conservar NaN: solo se usan en EDA, nunca en el modelo (son posteriores al partido)"),
        ("Cuotas Bet365 faltantes", m["OddHome"].isna(), "Completitud", "Excluir del modelo y de la simulación de apuestas"),
        ("Columnas de clúster C_* del autor del espejo (calculadas con estadísticas del propio partido)",
         m["C_LTH"].notna(), "Fuga de información", "Eliminar las 6 columnas"),
        ("Partidos sin resultado (suspendidos o no jugados, p. ej. Chapecoense 2016)",
         m["FTHome"].isna() | m["FTAway"].isna() | m["FTResult"].isna(), "Completitud", "Eliminar el partido"),
        ("Filas duplicadas exactas", m.duplicated(), "Unicidad", "Eliminar duplicados"),
        ("Partidos duplicados (misma liga, fecha, local y visitante)",
         m.duplicated(["Division", "MatchDate", "HomeTeam", "AwayTeam"]), "Unicidad", "Eliminar duplicados"),
    ]
    filas = []
    for problema, mascara, tipo, accion in problemas:
        n = int(mascara.fillna(False).sum())
        filas.append({"problema": problema, "tipo": tipo, "registros": n, "porcentaje": round(100 * n / len(m), 3), "accion": accion})
    return pd.DataFrame(filas)


def diagnostico_transfermarkt(tm_valores: pd.DataFrame, tm_partidos: pd.DataFrame) -> pd.DataFrame:
    """Problemas de calidad de las tablas de Transfermarkt."""
    filas = [
        ("Valoraciones con valor de mercado 0 €", int((tm_valores["market_value_in_eur"] <= 0).sum()), "Validez", "Excluir de la suma de plantilla"),
        ("Valoraciones duplicadas (jugador, fecha)", int(tm_valores.duplicated(["player_id", "date"]).sum()), "Unicidad", "Conservar la última"),
        ("Valoraciones con club 'Unknown'", int((tm_valores["current_club_name"] == "Unknown").sum()), "Completitud", "Se usa el id del club, no el nombre"),
        ("Partidos de Transfermarkt duplicados (game_id)", int(tm_partidos["game_id"].duplicated().sum()), "Unicidad", "Eliminar duplicados"),
        ("Partidos de Transfermarkt sin nombre de club", int(tm_partidos[["home_club_name", "away_club_name"]].isna().any(axis=1).sum()), "Completitud", "Se usa el id del club"),
    ]
    return pd.DataFrame(filas, columns=["problema", "registros", "tipo", "accion"])


# ==================================================================================================
# 1. LIMPIEZA
# ==================================================================================================
COLUMNAS_FUGA = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]


def limpiar_partidos(m: pd.DataFrame, reg: RegistroETL) -> pd.DataFrame:
    m = m.copy()
    n = len(m)

    # 1.1 espacios en nombres
    afectados = int(((m["HomeTeam"] != m["HomeTeam"].str.strip()) | (m["AwayTeam"] != m["AwayTeam"].str.strip())).sum())
    m["HomeTeam"] = m["HomeTeam"].str.strip()
    m["AwayTeam"] = m["AwayTeam"].str.strip()
    reg.anotar("Limpieza", "Recortar espacios en nombres de equipo", n, len(m), afectados, "str.strip() sobre local y visitante")

    # 1.2 duplicados exactos y por clave natural
    antes = len(m)
    m = m.drop_duplicates()
    m = m.drop_duplicates(["Division", "MatchDate", "HomeTeam", "AwayTeam"], keep="last")
    reg.anotar("Limpieza", "Eliminar duplicados (exactos y por liga+fecha+equipos)", antes, len(m), antes - len(m),
               "Se verificó la clave natural del partido")

    # 1.2b partidos sin resultado (suspendidos / no jugados): no sirven para EDA ni para el modelo
    antes = len(m)
    m = m.dropna(subset=["FTHome", "FTAway", "FTResult"])
    reg.anotar("Limpieza", "Eliminar partidos sin resultado (suspendidos o no jugados)", antes, len(m), antes - len(m),
               "Ej.: Chapecoense–Atlético MG (nov-2016, tragedia aérea), Panathinaikos–Olympiakos 2019")
    n = len(m)

    # 1.3 cuotas 1X2 imposibles o con margen absurdo -> NaN en las tres
    cuotas = ["OddHome", "OddDraw", "OddAway"]
    sobre = (1 / m[cuotas]).sum(axis=1, min_count=3)
    malas = (m[cuotas] < 1.01).any(axis=1) | (sobre < 1) | (sobre > 1.3)
    m.loc[malas, cuotas] = np.nan
    reg.anotar("Limpieza", "Anular cuotas 1X2 imposibles (<1,01) o con margen fuera de [0 %, 30 %]", n, len(m), int(malas.sum()),
               "Las 3 cuotas del partido pasan a NaN")

    # 1.4 cuotas máximas inconsistentes
    maximas = ["MaxHome", "MaxDraw", "MaxAway"]
    ratio = pd.concat([m["MaxHome"] / m["OddHome"], m["MaxDraw"] / m["OddDraw"], m["MaxAway"] / m["OddAway"]], axis=1).max(axis=1)
    malas = (ratio > 3) | (m[maximas] < 1.01).any(axis=1)
    m.loc[malas, maximas] = np.nan
    reg.anotar("Limpieza", "Anular cuotas máximas >3× Bet365 o <1,01", n, len(m), int(malas.sum()), "Error de captura en el mercado")

    # 1.5 cuotas de goles y hándicap fuera de rango
    goles = ["Over25", "Under25", "MaxOver25", "MaxUnder25"]
    malas = (m[goles] > 20).any(axis=1) | (m[goles] < 1.01).any(axis=1)
    m.loc[malas, goles] = np.nan
    afect_goles = int(malas.sum())
    handi = ["HandiSize", "HandiHome", "HandiAway"]
    malas = (m["HandiSize"].abs() > 10) | ~m["HandiHome"].between(1.01, 20) & m["HandiHome"].notna() \
        | ~m["HandiAway"].between(1.01, 20) & m["HandiAway"].notna()
    m.loc[malas, handi] = np.nan
    reg.anotar("Limpieza", "Anular cuotas de goles (>20) y hándicaps imposibles (|h|>10, p. ej. −99,9)", n, len(m),
               afect_goles + int(malas.sum()), "Valores imposibles en el mercado de goles y asiático")

    # 1.6 estadísticas imposibles
    tiros = ["HomeShots", "AwayShots", "HomeTarget", "AwayTarget"]
    malas_t = (m["HomeTarget"] > m["HomeShots"]) | (m["AwayTarget"] > m["AwayShots"])
    m.loc[malas_t, tiros] = np.nan
    malas_f = (m["HomeFouls"] > 60) | (m["AwayFouls"] > 60)
    m.loc[malas_f, ["HomeFouls", "AwayFouls"]] = np.nan
    malas_r = (m["HomeRed"] > 5) | (m["AwayRed"] > 5)
    m.loc[malas_r, ["HomeRed", "AwayRed"]] = np.nan
    reg.anotar("Limpieza", "Anular estadísticas imposibles (tiros al arco > tiros, >60 faltas, >5 rojas)", n, len(m),
               int(malas_t.sum() + malas_f.sum() + malas_r.sum()), "Se conserva el partido; solo la estadística pasa a NaN")

    # 1.7 columnas con fuga de información
    columnas_antes = m.shape[1]
    m = m.drop(columns=COLUMNAS_FUGA)
    reg.anotar("Limpieza", "Eliminar columnas C_* (clústeres calculados con estadísticas del mismo partido)", columnas_antes,
               m.shape[1], len(COLUMNAS_FUGA), "Conteo en columnas: usarlas filtraría el resultado al modelo")
    return m


# ==================================================================================================
# 2. NORMALIZACIÓN
# ==================================================================================================
RENOMBRAR = {
    "Division": "liga_codigo", "MatchDate": "fecha", "MatchTime": "hora", "HomeTeam": "local", "AwayTeam": "visitante",
    "HomeElo": "elo_clubelo_local", "AwayElo": "elo_clubelo_visitante",
    "Form3Home": "forma3_local", "Form5Home": "forma5_local", "Form3Away": "forma3_visitante", "Form5Away": "forma5_visitante",
    "FTHome": "goles_local", "FTAway": "goles_visitante", "FTResult": "resultado",
    "HTHome": "goles_ht_local", "HTAway": "goles_ht_visitante", "HTResult": "resultado_ht",
    "HomeShots": "tiros_local", "AwayShots": "tiros_visitante", "HomeTarget": "tiros_arco_local", "AwayTarget": "tiros_arco_visitante",
    "HomeFouls": "faltas_local", "AwayFouls": "faltas_visitante", "HomeCorners": "corners_local", "AwayCorners": "corners_visitante",
    "HomeYellow": "amarillas_local", "AwayYellow": "amarillas_visitante", "HomeRed": "rojas_local", "AwayRed": "rojas_visitante",
    "OddHome": "cuota_local", "OddDraw": "cuota_empate", "OddAway": "cuota_visitante",
    "MaxHome": "cuota_max_local", "MaxDraw": "cuota_max_empate", "MaxAway": "cuota_max_visitante",
    "Over25": "cuota_mas25", "Under25": "cuota_menos25", "MaxOver25": "cuota_max_mas25", "MaxUnder25": "cuota_max_menos25",
    "HandiSize": "handicap", "HandiHome": "cuota_handicap_local", "HandiAway": "cuota_handicap_visitante",
}
CONTEOS = [c for c in RENOMBRAR.values() if c.startswith(("goles", "tiros", "faltas", "corners", "amarillas", "rojas", "forma"))]


def normalizar_partidos(m: pd.DataFrame, reg: RegistroETL) -> pd.DataFrame:
    memoria_antes = memoria_mb(m)
    m = m.rename(columns=RENOMBRAR)
    reg.anotar("Normalización", "Renombrar 42 columnas a español snake_case", len(m), len(m), len(RENOMBRAR),
               "HomeTeam → local, OddHome → cuota_local, FTResult → resultado…")

    # fechas y hora
    m["fecha"] = pd.to_datetime(m["fecha"], format="%Y-%m-%d")
    m["hora"] = m["hora"].str.slice(0, 5)
    m = m.sort_values(["fecha", "liga_codigo", "local"], kind="stable").reset_index(drop=True)

    # resultado en español (H/D/A → Local/Empate/Visitante)
    m["resultado"] = m["resultado"].map(C.RESULTADOS)
    m["resultado_ht"] = m["resultado_ht"].map(C.RESULTADOS)

    # dimensiones de liga (país, nivel, continente, nombre)
    dims = pd.DataFrame.from_dict(C.LIGAS, orient="index",
                                  columns=["pais", "nivel", "continente", "liga", "tm_competicion", "calendario_anual"])
    m = m.join(dims, on="liga_codigo")

    # temporada unificada: europeas jul–jun ("2023/24"), calendario anual ("2023")
    anio = m["fecha"].dt.year
    europea_inicio = np.where(m["fecha"].dt.month >= 7, anio, anio - 1)
    m["temporada_inicio"] = np.where(m["calendario_anual"], anio, europea_inicio).astype("int16")
    etiqueta_europea = m["temporada_inicio"].astype(str) + "/" + ((m["temporada_inicio"] + 1) % 100).astype(str).str.zfill(2)
    m["temporada"] = np.where(m["calendario_anual"], m["temporada_inicio"].astype(str), etiqueta_europea)
    m["decada"] = (m["temporada_inicio"] // 10 * 10).astype(str) + "s"
    m["mes"] = m["fecha"].dt.month.astype("int8")
    reg.anotar("Normalización", "Unificar temporada (jul–jun vs. año natural) y añadir país/nivel/continente", len(m), len(m),
               int(m["calendario_anual"].sum()), "Afectados = partidos de ligas con calendario anual")

    # cuotas → probabilidades implícitas sin margen (normalización proporcional)
    inv = 1 / m[["cuota_local", "cuota_empate", "cuota_visitante"]]
    sobre = inv.sum(axis=1, min_count=3)
    m["margen_casa"] = sobre - 1
    m["prob_casa_local"] = inv["cuota_local"] / sobre
    m["prob_casa_empate"] = inv["cuota_empate"] / sobre
    m["prob_casa_visitante"] = inv["cuota_visitante"] / sobre
    inv_goles = 1 / m[["cuota_mas25", "cuota_menos25"]]
    m["prob_casa_mas25"] = inv_goles["cuota_mas25"] / inv_goles.sum(axis=1, min_count=2)
    reg.anotar("Normalización", "Convertir cuotas en probabilidades implícitas sin margen (suman 1)", len(m), len(m),
               int(sobre.notna().sum()), "p_i = (1/cuota_i) / Σ(1/cuota)")

    # tipos optimizados
    for col in ["liga_codigo", "pais", "continente", "liga", "tm_competicion", "resultado", "resultado_ht", "temporada", "decada", "local", "visitante"]:
        m[col] = m[col].astype("category")
    for col in CONTEOS:
        m[col] = m[col].astype("Int8") if m[col].max() < 120 else m[col].astype("Int16")
    m["nivel"] = m["nivel"].astype("int8")
    flotantes = m.select_dtypes("float64").columns
    m[flotantes] = m[flotantes].astype("float32")
    reg.anotar("Normalización", f"Optimizar tipos (category, Int8, float32): {memoria_antes:.0f} MB → {memoria_mb(m):.0f} MB",
               len(m), len(m), m.shape[1], "Conteo en columnas convertidas; la memoria se mide con deep=True")
    return m


# ==================================================================================================
# 4. ENRIQUECIMIENTO (va antes de la agregación porque los cubos usan las variables nuevas)
# ==================================================================================================
def calcular_elo_propio(m: pd.DataFrame, k: float = 20.0, ventaja_local: float = 65.0) -> pd.DataFrame:
    """Elo previo al partido para TODAS las ligas (ClubElo no cubre 37 % de los partidos).

    Fórmula tipo World Football Elo: K=20, ventaja de localía de 65 puntos y multiplicador por diferencia de goles.
    Los equipos nuevos arrancan en 1500 − 75·(nivel−1). El valor guardado es el previo al partido (sin fuga).
    """
    rating: dict[str, float] = {}
    elo_l = np.empty(len(m), dtype="float32")
    elo_v = np.empty(len(m), dtype="float32")
    locales = m["local"].astype(str).to_numpy()
    visitantes = m["visitante"].astype(str).to_numpy()
    gl = m["goles_local"].to_numpy(dtype="float64")
    gv = m["goles_visitante"].to_numpy(dtype="float64")
    niveles = m["nivel"].to_numpy()
    for i in range(len(m)):
        inicial = 1500.0 - 75.0 * (niveles[i] - 1)
        rl = rating.get(locales[i], inicial)
        rv = rating.get(visitantes[i], inicial)
        elo_l[i], elo_v[i] = rl, rv
        esperado_l = 1.0 / (1.0 + 10 ** ((rv - rl - ventaja_local) / 400.0))
        real_l = 1.0 if gl[i] > gv[i] else (0.5 if gl[i] == gv[i] else 0.0)
        dif = abs(gl[i] - gv[i])
        multiplicador = 1.0 if dif <= 1 else (1.5 if dif == 2 else (11.0 + dif) / 8.0)
        cambio = k * multiplicador * (real_l - esperado_l)
        rating[locales[i]] = rl + cambio
        rating[visitantes[i]] = rv - cambio
    m = m.copy()
    m["elo_propio_local"] = elo_l
    m["elo_propio_visitante"] = elo_v
    m["diferencia_elo"] = (elo_l - elo_v).astype("float32")
    m.attrs["elo_final"] = rating
    return m


def construir_formato_largo(m: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido (el doble de filas), base para la forma reciente y el cubo de equipos."""
    comunes = ["fecha", "liga_codigo", "liga", "pais", "nivel", "temporada", "temporada_inicio"]
    def lado(es_local: bool) -> pd.DataFrame:
        yo, rival = ("local", "visitante") if es_local else ("visitante", "local")
        t = m[comunes].copy()
        t["partido_id"] = m.index
        t["equipo"] = m[yo].astype(str)
        t["rival"] = m[rival].astype(str)
        t["es_local"] = es_local
        t["gf"] = m[f"goles_{yo}"].astype("float32")
        t["gc"] = m[f"goles_{rival}"].astype("float32")
        t["elo_propio"] = m[f"elo_propio_{yo}"]
        return t
    largo = pd.concat([lado(True), lado(False)], ignore_index=True)
    largo["puntos"] = np.select([largo["gf"] > largo["gc"], largo["gf"] == largo["gc"]], [3, 1], 0).astype("int8")
    largo["resultado_equipo"] = np.select([largo["puntos"] == 3, largo["puntos"] == 1], ["Victoria", "Empate"], "Derrota")
    return largo.sort_values(["equipo", "fecha", "es_local"]).reset_index(drop=True)


def features_forma(largo: pd.DataFrame) -> pd.DataFrame:
    """Forma reciente y descanso. Versión *pre* (shift(1), sin fuga) para entrenar y *post* para predecir."""
    g = largo.groupby("equipo", sort=False)
    for col in ["gf", "gc", "puntos"]:
        largo[f"{col}_prom5_post"] = g[col].transform(lambda s: s.rolling(5, min_periods=1).mean()).astype("float32")
        largo[f"{col}_prom5_pre"] = largo.groupby("equipo", sort=False)[f"{col}_prom5_post"].shift(1)
    # puntos por partido en la temporada (misma liga)
    gt = largo.groupby(["equipo", "temporada", "liga_codigo"], sort=False, observed=True)
    acumulado = gt["puntos"].cumsum()
    jugados = gt.cumcount() + 1
    largo["ppg_temporada_post"] = (acumulado / jugados).astype("float32")
    largo["ppg_temporada_pre"] = ((acumulado - largo["puntos"]) / (jugados - 1)).replace([np.inf, -np.inf], np.nan).astype("float32")
    largo["partidos_temporada_pre"] = (jugados - 1).astype("int16")
    # rendimiento como local (si es local) o como visitante (si es visitante) en los últimos 5 de esa condición
    gc_ = largo.groupby(["equipo", "es_local"], sort=False)
    largo["record_condicion_post"] = gc_["puntos"].transform(lambda s: s.rolling(5, min_periods=1).sum()).astype("float32")
    largo["record_condicion_pre"] = largo.groupby(["equipo", "es_local"], sort=False)["record_condicion_post"].shift(1)
    # días de descanso desde el partido anterior (recortado a 30: después del receso deja de ser informativo)
    largo["dias_descanso"] = g["fecha"].diff().dt.days.clip(upper=30).astype("float32")
    return largo


def unir_forma_a_partidos(m: pd.DataFrame, largo: pd.DataFrame) -> pd.DataFrame:
    columnas = ["gf_prom5_pre", "gc_prom5_pre", "puntos_prom5_pre", "ppg_temporada_pre", "partidos_temporada_pre",
                "record_condicion_pre", "dias_descanso"]
    for es_local, sufijo in [(True, "local"), (False, "visitante")]:
        parte = largo.loc[largo["es_local"] == es_local, ["partido_id", *columnas]].set_index("partido_id")
        parte.columns = [f"{c.removesuffix('_pre')}_{sufijo}" for c in columnas]
        m = m.join(parte)
    m["dif_puntos_prom5"] = (m["puntos_prom5_local"] - m["puntos_prom5_visitante"]).astype("float32")
    m["dif_ppg_temporada"] = (m["ppg_temporada_local"] - m["ppg_temporada_visitante"]).astype("float32")
    return m


def mapear_transfermarkt(m: pd.DataFrame, tm_partidos: pd.DataFrame, tm_clubes: pd.DataFrame) -> pd.DataFrame:
    """Homologa nombres de equipo Football-Data ↔ club_id de Transfermarkt.

    Los nombres no coinciden ("Man United" vs "Manchester United Football Club"), así que se usa evidencia de
    partidos: se cruzan los partidos de ambas fuentes por liga, fecha (±1 día por zona horaria) y marcador, y a
    cada nombre se le asigna el club_id con el que más veces coincide. La similitud de texto (rapidfuzz) queda
    como columna de control.
    """
    from rapidfuzz import fuzz

    codigos = {cod: info[4] for cod, info in C.LIGAS.items() if info[4]}
    fd = m.loc[m["liga_codigo"].isin(codigos.keys()), ["liga_codigo", "fecha", "local", "visitante", "goles_local", "goles_visitante"]].copy()
    fd["tm_competicion"] = fd["liga_codigo"].astype(str).map(codigos)
    fd[["local", "visitante"]] = fd[["local", "visitante"]].astype(str)
    fd[["goles_local", "goles_visitante"]] = fd[["goles_local", "goles_visitante"]].astype("int64")

    tm = tm_partidos.drop_duplicates("game_id")
    tm = tm.loc[tm["competition_id"].isin(codigos.values()),
                ["competition_id", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]].copy()
    tm["date"] = pd.to_datetime(tm["date"])
    cruces = []
    for desfase in (-1, 0, 1):
        t = tm.assign(fecha=tm["date"] + pd.Timedelta(days=desfase))
        cruces.append(fd.merge(t, left_on=["tm_competicion", "fecha", "goles_local", "goles_visitante"],
                               right_on=["competition_id", "fecha", "home_club_goals", "away_club_goals"]))
    cruce = pd.concat(cruces, ignore_index=True)
    votos = pd.concat([
        cruce[["local", "home_club_id"]].set_axis(["equipo", "club_id"], axis=1),
        cruce[["visitante", "away_club_id"]].set_axis(["equipo", "club_id"], axis=1),
    ])
    conteo = votos.value_counts().rename("coincidencias").reset_index()
    conteo["total_equipo"] = conteo.groupby("equipo")["coincidencias"].transform("sum")
    conteo["participacion"] = conteo["coincidencias"] / conteo["total_equipo"]
    mejor = conteo.sort_values("coincidencias", ascending=False).drop_duplicates("equipo")
    # un club_id solo puede pertenecer a un nombre: se queda el de más coincidencias
    mejor = mejor.sort_values("coincidencias", ascending=False).drop_duplicates("club_id")
    mejor = mejor[(mejor["coincidencias"] >= 5) & (mejor["participacion"] >= 0.30)]
    nombres = tm_clubes.set_index("club_id")["name"]
    mejor["nombre_transfermarkt"] = mejor["club_id"].map(nombres)
    mejor["similitud_texto"] = [
        fuzz.token_set_ratio(a, b) if isinstance(b, str) else np.nan
        for a, b in zip(mejor["equipo"], mejor["nombre_transfermarkt"])
    ]
    return mejor.sort_values("equipo").reset_index(drop=True)


def valor_plantillas(tm_valores: pd.DataFrame) -> pd.DataFrame:
    """Valor de mercado de cada plantilla al inicio de cada temporada (solo información previa: sin fuga).

    Para el corte de una temporada (1 de agosto en ligas europeas, 1 de febrero en ligas de año natural) se toma la
    última valoración de cada jugador en los 12 meses previos y se suma por club.
    """
    v = tm_valores[tm_valores["market_value_in_eur"] > 0].copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v.sort_values("date").drop_duplicates(["player_id", "date"], keep="last")
    salida = []
    for anio in range(2004, 2027):
        for anual, corte in [(False, pd.Timestamp(anio, 8, 1)), (True, pd.Timestamp(anio, 2, 1))]:
            ventana = v[(v["date"] < corte) & (v["date"] >= corte - pd.Timedelta(days=365))]
            ultima = ventana.drop_duplicates("player_id", keep="last")
            suma = ultima.groupby("current_club_id").agg(valor_plantilla_eur=("market_value_in_eur", "sum"),
                                                          jugadores_valorados=("player_id", "size"))
            suma["temporada_inicio"] = anio
            suma["calendario_anual"] = anual
            salida.append(suma.reset_index().rename(columns={"current_club_id": "club_id"}))
    return pd.concat(salida, ignore_index=True)


def unir_valor_mercado(m: pd.DataFrame, mapeo: pd.DataFrame, valores: pd.DataFrame) -> pd.DataFrame:
    ids = mapeo.set_index("equipo")["club_id"]
    clave = valores.set_index(["club_id", "temporada_inicio", "calendario_anual"])["valor_plantilla_eur"]
    for lado in ("local", "visitante"):
        club = m[lado].astype(str).map(ids)
        indice = pd.MultiIndex.from_arrays([club, m["temporada_inicio"].astype(int), m["calendario_anual"].astype(bool)])
        m[f"valor_plantilla_{lado}"] = clave.reindex(indice).to_numpy(dtype="float64") / 1e6  # millones de euros
        m[f"valor_plantilla_{lado}"] = m[f"valor_plantilla_{lado}"].astype("float32")
    m["log_ratio_valor"] = (np.log1p(m["valor_plantilla_local"]) - np.log1p(m["valor_plantilla_visitante"])).astype("float32")
    return m


def _confederacion_selecciones(sel: pd.DataFrame) -> pd.Series:
    """Confederación de cada selección según el torneo continental que más juega (Euro → UEFA, Copa América → CONMEBOL…)."""
    reglas = [("UEFA", "UEFA"), ("Copa América", "CONMEBOL"), ("CONMEBOL", "CONMEBOL"), ("Gold Cup", "CONCACAF"),
              ("CONCACAF", "CONCACAF"), ("African", "CAF"), ("CAF", "CAF"), ("AFC", "AFC"), ("Asian", "AFC"), ("Oceania", "OFC"), ("OFC", "OFC")]
    largo = pd.concat([sel[["home_team", "tournament"]].set_axis(["equipo", "torneo"], axis=1),
                       sel[["away_team", "tournament"]].set_axis(["equipo", "torneo"], axis=1)])
    largo["confederacion"] = None
    for clave, conf in reglas:
        sin = largo["confederacion"].isna() & largo["torneo"].str.contains(clave, regex=False)
        largo.loc[sin, "confederacion"] = conf
    return largo.dropna().groupby("equipo")["confederacion"].agg(lambda s: s.value_counts().index[0])


def variables_fatiga(m: pd.DataFrame, mapeo: pd.DataFrame, tm_partidos: pd.DataFrame, apariciones: pd.DataFrame,
                     jugadores: pd.DataFrame, selecciones: pd.DataFrame, reg: RegistroETL) -> pd.DataFrame:
    """Fatiga y parones de selecciones, por equipo y partido, solo con información previa al partido.

    - descanso_real: días desde el último partido del club en CUALQUIER competición (liga, Champions, copa).
    - carga_min14: minutos jugados por los jugadores del club entre 15 y 2 días antes del partido.
    - post_paron: entre el partido anterior del club y este hubo una fecha FIFA (≥ 20 partidos de selecciones).
    - convocados: jugadores habituales (≥ 45 min en algún partido de los 35 días previos al parón) que jugaron con
      su selección en ese parón; viajeros: los de esos convocados cuya selección es de otra confederación que la liga.
    Solo hay datos de Transfermarkt para ligas con código Transfermarkt desde la temporada 2012: fuera de eso, NaN.
    """
    ids = mapeo.set_index("equipo")["club_id"]
    cubierto = m["tm_competicion"].notna() & (m["temporada_inicio"] >= 2012)
    # calendario de cada club en todas las competiciones de Transfermarkt
    tm = tm_partidos.drop_duplicates("game_id")
    calendario = pd.concat([tm[["home_club_id", "date"]].set_axis(["club_id", "fecha_prev"], axis=1),
                            tm[["away_club_id", "date"]].set_axis(["club_id", "fecha_prev"], axis=1)])
    calendario["fecha_prev"] = pd.to_datetime(calendario["fecha_prev"])
    # + partidos de liga de Football-Data: cubren lo que Transfermarkt aún no tiene (p. ej. la temporada 2026/27)
    liga = pd.concat([pd.DataFrame({"club_id": m[l].astype(str).map(ids), "fecha_prev": m["fecha"]}) for l in ("local", "visitante")])
    calendario = pd.concat([calendario, liga.dropna()]).astype({"club_id": "int64"})
    calendario = calendario.drop_duplicates().sort_values("fecha_prev")
    fin_minutos = pd.to_datetime(apariciones["date"]).max()     # después de esta fecha no hay minutos: NaN, no 0
    # minutos acumulados por club y día
    ap = apariciones.assign(fecha=pd.to_datetime(apariciones["date"]))
    minutos = ap.groupby(["player_club_id", "fecha"])["minutes_played"].sum().rename("min").reset_index()
    minutos = minutos.sort_values(["player_club_id", "fecha"])
    minutos["acum"] = minutos.groupby("player_club_id")["min"].cumsum()
    minutos = minutos.rename(columns={"player_club_id": "club_id"}).sort_values("fecha")
    # selecciones: partidos por día y por equipo (desde 2011 para cubrir la primera temporada)
    sel = selecciones.assign(fecha=pd.to_datetime(selecciones["date"]))
    sel = sel[sel["fecha"] >= "2011-06-01"]
    conf_sel = _confederacion_selecciones(sel)
    por_dia = sel.groupby("fecha").size().rename("n").reset_index().sort_values("fecha")
    por_dia["acum"] = por_dia["n"].cumsum()
    partidos_sel = pd.concat([sel[["home_team", "fecha"]].set_axis(["seleccion", "fecha_sel"], axis=1),
                              sel[["away_team", "fecha"]].set_axis(["seleccion", "fecha_sel"], axis=1)])
    nac = jugadores.assign(seleccion=jugadores["country_of_citizenship"].replace(C.ALIAS_SELECCION))
    # shortcut: international_caps es el dato actual del jugador (no el de esa fecha); aproxima "es internacional".
    nac = nac[nac["international_caps"].fillna(0) >= 1][["player_id", "seleccion"]]

    def asof(izq, der, col_izq, col_der, por=None):
        return pd.merge_asof(izq.sort_values(col_izq), der, left_on=col_izq, right_on=col_der, by=por, direction="backward")

    for lado in ("local", "visitante"):
        base = pd.DataFrame({"idx": m.index, "club_id": m[lado].astype(str).map(ids), "fecha": m["fecha"]})
        base = base[cubierto & base["club_id"].notna()].copy()
        base["club_id"] = base["club_id"].astype("int64")
        base["ref"] = base["fecha"] - pd.Timedelta(days=2)         # el mismo partido puede figurar con ±1 día
        base = asof(base, calendario.astype({"club_id": "int64"}), "ref", "fecha_prev", "club_id")
        base["descanso_real"] = (base["fecha"] - base["fecha_prev"]).dt.days.clip(upper=30)
        # carga de minutos: acumulado(d−2) − acumulado(d−15)
        mins = minutos.astype({"club_id": "int64"})
        a2 = asof(base[["idx", "club_id", "ref"]], mins[["club_id", "fecha", "acum"]], "ref", "fecha", "club_id")
        base["ref15"] = base["fecha"] - pd.Timedelta(days=15)
        a15 = asof(base[["idx", "club_id", "ref15"]], mins[["club_id", "fecha", "acum"]], "ref15", "fecha", "club_id")
        base = base.merge(a2[["idx", "acum"]].rename(columns={"acum": "a2"}), on="idx", how="left")
        base = base.merge(a15[["idx", "acum"]].rename(columns={"acum": "a15"}), on="idx", how="left")
        base["carga_min14"] = (base["a2"].fillna(0) - base["a15"].fillna(0)).where(base["a2"].notna() & (base["fecha"] <= fin_minutos))
        # ¿hubo fecha FIFA entre el partido anterior y este?
        hoy = asof(base[["idx", "fecha"]].assign(r=base["fecha"] - pd.Timedelta(days=1)), por_dia[["fecha", "acum"]].rename(columns={"fecha": "f"}), "r", "f")
        antes = asof(base[["idx", "fecha_prev"]].dropna(), por_dia[["fecha", "acum"]].rename(columns={"fecha": "f"}), "fecha_prev", "f")
        base = base.merge(hoy[["idx", "acum"]].rename(columns={"acum": "s1"}), on="idx", how="left")
        base = base.merge(antes[["idx", "acum"]].rename(columns={"acum": "s0"}), on="idx", how="left")
        base["post_paron"] = (((base["s1"].fillna(0) - base["s0"].fillna(0)) >= 20) & base["fecha_prev"].notna()
                              & (base["descanso_real"] <= 21))
        # convocados y viajeros (solo partidos después de un parón)
        pp = base.loc[base["post_paron"], ["idx", "club_id", "fecha_prev", "fecha"]]
        habituales = pp.merge(ap.loc[ap["minutes_played"] >= 45, ["player_club_id", "player_id", "fecha"]],
                              left_on="club_id", right_on="player_club_id", suffixes=("", "_ap"))
        habituales = habituales[(habituales["fecha_ap"] <= habituales["fecha_prev"]) &
                                (habituales["fecha_ap"] >= habituales["fecha_prev"] - pd.Timedelta(days=35))]
        habituales = habituales.drop_duplicates(["idx", "player_id"]).merge(nac, on="player_id")
        jugo = habituales.merge(partidos_sel, on="seleccion")
        jugo = jugo[(jugo["fecha_sel"] > jugo["fecha_prev"]) & (jugo["fecha_sel"] < jugo["fecha"])].drop_duplicates(["idx", "player_id"])
        conf_liga = m.loc[jugo["idx"], "pais"].astype(str).map(C.CONFEDERACION_PAIS).fillna("UEFA").to_numpy()
        jugo["viajero"] = jugo["seleccion"].map(conf_sel).to_numpy() != conf_liga
        conteo = jugo.groupby("idx").agg(convocados=("player_id", "size"), viajeros=("viajero", "sum"))
        base = base.set_index("idx").join(conteo)
        base.loc[base["post_paron"], ["convocados", "viajeros"]] = base.loc[base["post_paron"], ["convocados", "viajeros"]].fillna(0)
        base.loc[~base["post_paron"], ["convocados", "viajeros"]] = 0
        base.loc[base["fecha_prev"] > fin_minutos, ["convocados", "viajeros"]] = np.nan
        for col in ["descanso_real", "carga_min14", "convocados", "viajeros"]:
            m[f"{col}_{lado}"] = base[col].reindex(m.index).astype("float32")
        m[f"post_paron_{lado}"] = base["post_paron"].reindex(m.index).astype("float32")
    m["post_paron"] = m[["post_paron_local", "post_paron_visitante"]].max(axis=1)
    m["dif_carga_min14"] = (m["carga_min14_local"] - m["carga_min14_visitante"]).astype("float32")
    m["dif_descanso_real"] = (m["descanso_real_local"] - m["descanso_real_visitante"]).astype("float32")
    m["dif_viajeros"] = (m["viajeros_local"] - m["viajeros_visitante"]).astype("float32")
    n = len(m)
    reg.anotar("Enriquecimiento", "Descanso real (todas las competiciones) y carga de minutos 14 días (Transfermarkt appearances)", n, n,
               int(m["descanso_real_local"].notna().sum()), f"{len(apariciones):,} apariciones de jugadores → minutos por club y día")
    reg.anotar("Enriquecimiento", "Parón de selecciones: post-parón, convocados y viajeros entre confederaciones (martj42 + players)", n, n,
               int((m["post_paron"] == 1).sum()), f"{len(selecciones):,} partidos de selecciones · {len(nac):,} internacionales con selección identificada")
    return m


def variables_derivadas(m: pd.DataFrame) -> pd.DataFrame:
    """Variables de negocio derivadas (para EDA, dashboard y modelo)."""
    m["total_goles"] = (m["goles_local"] + m["goles_visitante"]).astype("int8")
    m["mas_de_2_5"] = m["total_goles"] > 2.5
    m["ambos_marcan"] = (m["goles_local"] > 0) & (m["goles_visitante"] > 0)
    probs = m[["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"]].to_numpy()
    tiene = ~np.isnan(probs).any(axis=1)
    favorito = np.full(len(m), None, dtype=object)
    favorito[tiene] = np.array(C.ORDEN_RESULTADOS)[np.nanargmax(probs[tiene], axis=1)]
    m["favorito_casa"] = pd.Categorical(favorito, categories=C.ORDEN_RESULTADOS)
    m["acierto_favorito"] = (m["favorito_casa"].astype(object) == m["resultado"].astype(object)).where(tiene)
    prob_ganador = np.select([m["resultado"] == "Local", m["resultado"] == "Empate"],
                             [m["prob_casa_local"], m["prob_casa_empate"]], m["prob_casa_visitante"])
    m["prob_casa_resultado"] = prob_ganador.astype("float32")
    m["sorpresa"] = (m["prob_casa_resultado"] < 0.25).where(tiene)       # ganó algo que la casa daba < 25 %
    m["elo_imputado"] = m["elo_clubelo_local"].isna() | m["elo_clubelo_visitante"].isna()
    return m


# ==================================================================================================
# 3. AGREGACIÓN (cubos OLAP)
# ==================================================================================================
def cubo_liga_temporada(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["continente", "pais", "liga_codigo", "liga", "nivel", "temporada", "temporada_inicio"], observed=True)
    cubo = g.agg(
        partidos=("resultado", "size"),
        pct_local=("resultado", lambda s: (s == "Local").mean() * 100),
        pct_empate=("resultado", lambda s: (s == "Empate").mean() * 100),
        pct_visitante=("resultado", lambda s: (s == "Visitante").mean() * 100),
        goles_prom=("total_goles", "mean"),
        pct_mas25=("mas_de_2_5", "mean"),
        margen_casa_prom=("margen_casa", "mean"),
        acierto_favorito=("acierto_favorito", "mean"),
        pct_sorpresas=("sorpresa", "mean"),
        amarillas_prom=("amarillas_local", "mean"),
        valor_plantilla_prom=("valor_plantilla_local", "mean"),
    ).reset_index()
    for col in ["pct_mas25", "margen_casa_prom", "acierto_favorito", "pct_sorpresas"]:
        cubo[col] = cubo[col] * 100
    return cubo.round(3)


def cubo_pais_decada_nivel(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["continente", "pais", "decada", "nivel", "resultado"], observed=True)
    cubo = g.agg(partidos=("total_goles", "size"), goles_prom=("total_goles", "mean"),
                 prob_casa_prom=("prob_casa_resultado", "mean")).reset_index()
    cubo["pct_dentro_grupo"] = cubo["partidos"] / cubo.groupby(["pais", "decada", "nivel"], observed=True)["partidos"].transform("sum") * 100
    return cubo.round(3)


def cubo_equipo_temporada(m: pd.DataFrame, largo: pd.DataFrame) -> pd.DataFrame:
    g = largo.groupby(["equipo", "pais", "liga_codigo", "liga", "nivel", "temporada", "temporada_inicio"], observed=True)
    cubo = g.agg(partidos=("puntos", "size"), puntos=("puntos", "sum"), gf=("gf", "sum"), gc=("gc", "sum"),
                 victorias=("puntos", lambda s: (s == 3).sum()), empates=("puntos", lambda s: (s == 1).sum()),
                 elo_final=("elo_propio", "last")).reset_index()
    cubo["dif_goles"] = cubo["gf"] - cubo["gc"]
    cubo["derrotas"] = cubo["partidos"] - cubo["victorias"] - cubo["empates"]
    cubo["ppg"] = cubo["puntos"] / cubo["partidos"]
    valores = pd.concat([
        m[["local", "temporada", "valor_plantilla_local"]].set_axis(["equipo", "temporada", "valor_plantilla_meur"], axis=1),
        m[["visitante", "temporada", "valor_plantilla_visitante"]].set_axis(["equipo", "temporada", "valor_plantilla_meur"], axis=1),
    ])
    valores["equipo"] = valores["equipo"].astype(str)
    valores["temporada"] = valores["temporada"].astype(str)
    valores = valores.dropna().drop_duplicates(["equipo", "temporada"])
    cubo["temporada"] = cubo["temporada"].astype(str)
    cubo = cubo.merge(valores, on=["equipo", "temporada"], how="left")
    cubo["posicion"] = cubo.groupby(["liga_codigo", "temporada"], observed=True)["puntos"].rank(method="first", ascending=False)
    return cubo


# ==================================================================================================
# Estado actual de cada equipo (para el formulario de predicción del dashboard)
# ==================================================================================================
def estado_equipos(m: pd.DataFrame, largo: pd.DataFrame) -> pd.DataFrame:
    ultimo = largo.groupby("equipo").tail(1).set_index("equipo")
    estado = pd.DataFrame({
        "liga_codigo": ultimo["liga_codigo"].astype(str), "liga": ultimo["liga"].astype(str),
        "pais": ultimo["pais"].astype(str), "nivel": ultimo["nivel"],
        "ultima_fecha": ultimo["fecha"], "temporada": ultimo["temporada"].astype(str),
        "gf_prom5": ultimo["gf_prom5_post"], "gc_prom5": ultimo["gc_prom5_post"], "puntos_prom5": ultimo["puntos_prom5_post"],
        "ppg_temporada": ultimo["ppg_temporada_post"],
        "partidos_temporada": largo.groupby("equipo")["partidos_temporada_pre"].last() + 1,
    })
    elo_final = m.attrs.get("elo_final", {})
    estado["elo_propio"] = [elo_final.get(e, np.nan) for e in estado.index]
    # record como local y como visitante (últimos 5 en cada condición)
    for es_local, col in [(True, "record_local"), (False, "record_visitante")]:
        estado[col] = largo[largo["es_local"] == es_local].groupby("equipo")["record_condicion_post"].last()
    # forma3/forma5 al estilo de la fuente (puntos en los últimos 3 y 5 partidos)
    estado["forma3"] = largo.groupby("equipo")["puntos"].apply(lambda s: s.tail(3).sum())
    estado["forma5"] = largo.groupby("equipo")["puntos"].apply(lambda s: s.tail(5).sum())
    # último valor de plantilla conocido y último Elo de ClubElo
    for lado in ("local", "visitante"):
        ult = m.dropna(subset=[f"valor_plantilla_{lado}"]).groupby(lado, observed=True)[f"valor_plantilla_{lado}"].last()
        ult.index = ult.index.astype(str)
        estado[f"_valor_{lado}"] = ult
    estado["valor_plantilla"] = estado["_valor_local"].fillna(estado["_valor_visitante"])
    estado = estado.drop(columns=["_valor_local", "_valor_visitante"])
    for lado in ("local", "visitante"):
        ult = m.dropna(subset=[f"elo_clubelo_{lado}"]).groupby(lado, observed=True)[f"elo_clubelo_{lado}"].last()
        ult.index = ult.index.astype(str)
        estado[f"_elo_{lado}"] = ult
    estado["elo_clubelo"] = estado["_elo_local"].fillna(estado["_elo_visitante"])
    estado = estado.drop(columns=["_elo_local", "_elo_visitante"])
    estado.index.name = "equipo"
    return estado.reset_index()


# ==================================================================================================
# Orquestación
# ==================================================================================================
def ejecutar_etl(verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    reg = RegistroETL()
    crudos = cargar_crudos()
    t_carga = time.perf_counter() - t0
    m_crudo = crudos["partidos"]
    perfil_crudo = {
        nombre: {"filas": int(len(df)), "columnas": int(df.shape[1]), "memoria_mb": memoria_mb(df)}
        for nombre, df in crudos.items()
    }
    diag = diagnostico_calidad(m_crudo)
    diag_tm = diagnostico_transfermarkt(crudos["tm_valores"], crudos["tm_partidos"])

    m = limpiar_partidos(m_crudo, reg)
    m = normalizar_partidos(m, reg)

    # --- Enriquecimiento ---
    n = len(m)
    faltaba_elo = int((m["elo_clubelo_local"].isna() | m["elo_clubelo_visitante"].isna()).sum())
    m = calcular_elo_propio(m)
    # validación: lo que usa el modelo es la DIFERENCIA de Elo entre los dos equipos
    validos = m[["elo_clubelo_local", "elo_clubelo_visitante", "elo_propio_local", "elo_propio_visitante"]].dropna()
    corr_elo = float((validos["elo_clubelo_local"] - validos["elo_clubelo_visitante"]).corr(
        validos["elo_propio_local"] - validos["elo_propio_visitante"]))
    reg.anotar("Enriquecimiento", f"Elo propio previo al partido para todas las ligas (r = {corr_elo:.2f} con la diferencia de ClubElo)", n, n,
               faltaba_elo, "Afectados = partidos sin Elo de ClubElo que ahora tienen Elo")

    largo = features_forma(construir_formato_largo(m))
    m = unir_forma_a_partidos(m, largo)
    reg.anotar("Enriquecimiento", "Forma reciente (5 partidos), puntos por partido en la temporada, récord local/visitante y descanso",
               n, n, int(m["puntos_prom5_local"].notna().sum()), "Calculadas con shift(1): solo usan partidos anteriores")

    tm_valores = crudos["tm_valores"]
    mapeo = mapear_transfermarkt(m, crudos["tm_partidos"], crudos["tm_clubes"])
    equipos_ligas_tm = m.loc[m["tm_competicion"].notna(), "local"].astype(str).nunique()
    reg.anotar("Normalización", f"Homologar nombres Football-Data ↔ Transfermarkt ({len(mapeo)} de {equipos_ligas_tm} equipos de ligas con cobertura)",
               equipos_ligas_tm, len(mapeo), len(mapeo), "Voto por coincidencia de liga+fecha±1+marcador; similitud de texto como control")
    valores = valor_plantillas(tm_valores)
    m = unir_valor_mercado(m, mapeo, valores)
    con_valor = int(m["log_ratio_valor"].notna().sum())
    reg.anotar("Enriquecimiento", "Valor de mercado de ambas plantillas al inicio de la temporada (Transfermarkt)", n, n, con_valor,
               f"{len(tm_valores):,} valoraciones de jugadores → suma por club y temporada")

    m = variables_fatiga(m, mapeo, crudos["tm_partidos"], crudos["tm_apariciones"], crudos["tm_jugadores"],
                         crudos["selecciones"], reg)

    m = variables_derivadas(m)
    reg.anotar("Enriquecimiento", "Variables derivadas: total_goles, más de 2,5, ambos marcan, favorito, acierto, sorpresa", n, n, n,
               "Se añaden a todos los partidos")

    # --- Agregación ---
    cubo1 = cubo_liga_temporada(m)
    cubo2 = cubo_pais_decada_nivel(m)
    cubo3 = cubo_equipo_temporada(m, largo)
    reg.anotar("Agregación", "Cubo liga × temporada (13 medidas)", n, len(cubo1), len(cubo1), "Filas del cubo resultante")
    reg.anotar("Agregación", "Cubo país × década × nivel × resultado", n, len(cubo2), len(cubo2), "Filas del cubo resultante")
    reg.anotar("Agregación", "Cubo equipo × temporada (tabla de posiciones)", len(largo), len(cubo3), len(cubo3), "Filas del cubo resultante")

    estado = estado_equipos(m, largo)

    # --- ClubElo limpio (serie temporal para el dashboard) ---
    elo = crudos["elo"].rename(columns={"date": "fecha", "club": "equipo", "country": "pais_codigo", "elo": "elo"})
    elo["fecha"] = pd.to_datetime(elo["fecha"])
    elo = elo.drop_duplicates(["fecha", "equipo"])
    elo["equipo"] = elo["equipo"].astype("category")
    elo["pais_codigo"] = elo["pais_codigo"].astype("category")
    elo["elo"] = elo["elo"].astype("float32")

    # --- Guardado (Parquet) ---
    m.attrs = {}
    m.to_parquet(C.PARQUET_PARTIDOS, index=False)
    largo_guardar = largo[["partido_id", "fecha", "liga_codigo", "pais", "nivel", "temporada", "equipo", "rival", "es_local",
                           "gf", "gc", "puntos", "resultado_equipo", "elo_propio"]]
    largo_guardar.to_parquet(C.PARQUET_EQUIPOS, index=False)
    estado.to_parquet(C.PARQUET_ESTADO, index=False)
    elo.to_parquet(C.PARQUET_ELO, index=False)
    mapeo.to_parquet(C.PARQUET_MAPEO, index=False)
    cubo1.to_parquet(C.CUBO_LIGA_TEMPORADA, index=False)
    cubo2.to_parquet(C.CUBO_PAIS_DECADA, index=False)
    cubo3.to_parquet(C.CUBO_EQUIPO_TEMPORADA, index=False)

    resumen = {
        "perfil_crudo": perfil_crudo,
        "filas_crudas_total": int(sum(p["filas"] for p in perfil_crudo.values())),
        "tiempo_carga_s": round(t_carga, 2),
        "tiempo_total_s": round(time.perf_counter() - t0, 2),
        "partidos_limpios": int(len(m)), "columnas_limpias": int(m.shape[1]),
        "memoria_limpia_mb": memoria_mb(m),
        "peso_parquet_mb": round(C.PARQUET_PARTIDOS.stat().st_size / 1e6, 2),
        "filas_formato_largo": int(len(largo)),
        "equipos": int(estado.shape[0]), "ligas": int(m["liga_codigo"].nunique()),
        "fecha_min": str(m["fecha"].min().date()), "fecha_max": str(m["fecha"].max().date()),
        "correlacion_elo_propio_clubelo": round(corr_elo, 4),
        "equipos_mapeados_transfermarkt": int(len(mapeo)),
        "partidos_con_valor_mercado": con_valor,
        "diagnostico": diag.to_dict(orient="records"),
        "diagnostico_transfermarkt": diag_tm.to_dict(orient="records"),
        "pasos": reg.pasos,
    }
    C.JSON_ETL.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if verbose:
        print(reg.tabla().to_string())
        print(f"\nETL completo en {resumen['tiempo_total_s']} s → {C.PARQUET_PARTIDOS.name} "
              f"({resumen['partidos_limpios']:,} partidos, {resumen['peso_parquet_mb']} MB)")
    return {"resumen": resumen, "partidos": m, "largo": largo, "registro": reg, "diagnostico": diag,
            "diagnostico_tm": diag_tm, "mapeo": mapeo, "cubos": (cubo1, cubo2, cubo3), "estado": estado, "crudos": crudos}


if __name__ == "__main__":
    ejecutar_etl()
