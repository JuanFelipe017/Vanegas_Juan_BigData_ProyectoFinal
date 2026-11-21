"""Simulación de apuestas de valor (compartida por el pipeline, el notebook y el dashboard).

Regla: para cada partido se calcula el *valor esperado* de apostar 1 unidad a cada resultado,
``VE = p_modelo × cuota − 1``. Si el mejor VE supera el umbral, se apuesta 1 unidad a ese resultado.
Se usan las cuotas máximas del mercado (la mejor cuota disponible entre ~17 casas) y, si faltan, las de Bet365.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULTADOS = ["Local", "Empate", "Visitante"]


def cuotas_mercado(df: pd.DataFrame, usar_maximas: bool = True) -> np.ndarray:
    """Matriz n×3 de cuotas (local, empate, visitante): máximas con respaldo de Bet365."""
    bet365 = df[["cuota_local", "cuota_empate", "cuota_visitante"]].to_numpy(dtype="float64")
    if not usar_maximas:
        return bet365
    maximas = df[["cuota_max_local", "cuota_max_empate", "cuota_max_visitante"]].to_numpy(dtype="float64")
    return np.where(np.isnan(maximas), bet365, maximas)


def simular_apuestas(df: pd.DataFrame, columnas_prob: list[str], umbral_valor: float = 0.05,
                     usar_maximas: bool = True, cuota_maxima: float = 10.0) -> tuple[pd.DataFrame, dict]:
    """Devuelve (apuestas realizadas, resumen). ``df`` debe traer fecha, resultado, probabilidades y cuotas."""
    probs = df[columnas_prob].to_numpy(dtype="float64")
    cuotas = cuotas_mercado(df, usar_maximas)
    valor = probs * cuotas - 1
    valor[(cuotas > cuota_maxima) | np.isnan(cuotas)] = -np.inf      # evita cuotas extremas (varianza enorme)
    eleccion = np.argmax(valor, axis=1)
    mejor = valor[np.arange(len(df)), eleccion]
    apostar = mejor > umbral_valor
    ap = df.loc[apostar, ["fecha", "liga_codigo", "local", "visitante", "resultado"]].copy()
    ap["apuesta"] = np.array(RESULTADOS)[eleccion[apostar]]
    ap["cuota"] = cuotas[apostar, eleccion[apostar]]
    ap["prob_modelo"] = probs[apostar, eleccion[apostar]]
    ap["valor_esperado"] = mejor[apostar]
    ap["acierto"] = ap["apuesta"].to_numpy() == ap["resultado"].astype(str).to_numpy()
    ap["ganancia"] = np.where(ap["acierto"], ap["cuota"] - 1, -1.0)
    ap = ap.sort_values("fecha")
    ap["acumulado"] = ap["ganancia"].cumsum()
    n = len(ap)
    pico = ap["acumulado"].cummax().clip(lower=0) if n else pd.Series(dtype=float)  # la banca arranca en 0
    resumen = {
        "umbral_valor": umbral_valor,
        "partidos_evaluados": int(len(df)),
        "apuestas": int(n),
        "aciertos": int(ap["acierto"].sum()),
        "tasa_acierto": float(ap["acierto"].mean()) if n else 0.0,
        "cuota_media": float(ap["cuota"].mean()) if n else 0.0,
        "beneficio_unidades": float(ap["ganancia"].sum()) if n else 0.0,
        "roi": float(ap["ganancia"].sum() / n) if n else 0.0,
        "max_caida": float((pico - ap["acumulado"]).max()) if n else 0.0,
    }
    return ap, resumen


def curva_umbral(df: pd.DataFrame, columnas_prob: list[str], umbrales=None, usar_maximas: bool = True) -> pd.DataFrame:
    """ROI y número de apuestas para varios umbrales de valor esperado."""
    umbrales = np.round(np.arange(0.0, 0.31, 0.02), 2) if umbrales is None else umbrales
    filas = [simular_apuestas(df, columnas_prob, u, usar_maximas)[1] for u in umbrales]
    return pd.DataFrame(filas)
