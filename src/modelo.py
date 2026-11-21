"""Machine Learning (CRISP-DM: Modeling + Evaluation).

Tarea: **clasificación multiclase** del resultado de un partido (Local / Empate / Visitante) con información
disponible ANTES del pitazo inicial.

- Partición temporal (sin barajar): entrenamiento < 2022-07-01 ≤ validación < 2023-07-01 ≤ prueba.
- Líneas base: (1) "siempre gana el local" y (2) "gana el favorito de la casa de apuestas".
- Modelos: Regresión logística multinomial, Random Forest y LightGBM.
- Dos conjuntos de variables: **sin cuotas** (fútbol puro: Elo, forma, valor de plantilla…) y **con cuotas**
  (lo anterior + probabilidades implícitas del mercado).
- Se elige el mejor por log-loss en validación, se re-entrena con entrenamiento+validación y se evalúa en prueba.

Uso: ``python src/modelo.py``
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from apuestas import curva_umbral, simular_apuestas  # noqa: E402

CLASES = C.ORDEN_RESULTADOS  # ["Local", "Empate", "Visitante"]

VARS_FUTBOL = [
    "diferencia_elo", "elo_propio_local", "elo_propio_visitante",
    "puntos_prom5_local", "puntos_prom5_visitante", "dif_puntos_prom5",
    "gf_prom5_local", "gc_prom5_local", "gf_prom5_visitante", "gc_prom5_visitante",
    "ppg_temporada_local", "ppg_temporada_visitante", "dif_ppg_temporada", "partidos_temporada_local",
    "record_condicion_local", "record_condicion_visitante",
    "dias_descanso_local", "dias_descanso_visitante",
    "log_ratio_valor", "valor_plantilla_local", "valor_plantilla_visitante",
    "nivel", "mes",
]
VARS_MERCADO = ["prob_casa_local", "prob_casa_empate", "prob_casa_visitante", "logit_casa", "margen_casa", "prob_casa_mas25", "handicap"]
VARS_CATEGORICAS = ["liga_codigo"]
CONJUNTOS = {"sin_cuotas": VARS_FUTBOL, "con_cuotas": VARS_FUTBOL + VARS_MERCADO}

ETIQUETAS = {
    "diferencia_elo": "Diferencia de Elo (local − visitante)", "elo_propio_local": "Elo del local",
    "elo_propio_visitante": "Elo del visitante", "puntos_prom5_local": "Puntos prom. últimos 5 (local)",
    "puntos_prom5_visitante": "Puntos prom. últimos 5 (visitante)", "dif_puntos_prom5": "Diferencia de forma (5 partidos)",
    "gf_prom5_local": "Goles a favor prom. 5 (local)", "gc_prom5_local": "Goles en contra prom. 5 (local)",
    "gf_prom5_visitante": "Goles a favor prom. 5 (visitante)", "gc_prom5_visitante": "Goles en contra prom. 5 (visitante)",
    "ppg_temporada_local": "Puntos por partido en la temporada (local)",
    "ppg_temporada_visitante": "Puntos por partido en la temporada (visitante)",
    "dif_ppg_temporada": "Diferencia de puntos por partido en la temporada", "partidos_temporada_local": "Jornada (partidos jugados)",
    "record_condicion_local": "Puntos en últimos 5 como local", "record_condicion_visitante": "Puntos en últimos 5 como visitante",
    "dias_descanso_local": "Días de descanso (local)", "dias_descanso_visitante": "Días de descanso (visitante)",
    "log_ratio_valor": "Ratio de valor de plantilla (log)", "valor_plantilla_local": "Valor plantilla local (M€)",
    "valor_plantilla_visitante": "Valor plantilla visitante (M€)", "nivel": "Nivel de la liga", "mes": "Mes del partido",
    "prob_casa_local": "Prob. implícita casa: local", "prob_casa_empate": "Prob. implícita casa: empate",
    "prob_casa_visitante": "Prob. implícita casa: visitante", "margen_casa": "Margen de la casa",
    "prob_casa_mas25": "Prob. implícita casa: más de 2,5 goles", "logit_casa": "Log-odds casa (local vs. visitante)", "handicap": "Hándicap asiático", "liga_codigo": "Liga",
}


# ==================================================================================================
# Datos
# ==================================================================================================
def preparar_datos(partidos: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Filtra partidos con cuotas válidas (para comparar contra la casa) y hace la partición temporal."""
    if partidos is None:
        partidos = pd.read_parquet(C.PARQUET_PARTIDOS)
    df = partidos.dropna(subset=["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"]).copy()
    df["liga_codigo"] = df["liga_codigo"].astype(str)
    df["logit_casa"] = np.log(df["prob_casa_local"] / df["prob_casa_visitante"]).astype("float32")
    df["resultado"] = df["resultado"].astype(str)
    entrenamiento = df[df["fecha"] < C.FECHA_INICIO_VALIDACION]
    validacion = df[(df["fecha"] >= C.FECHA_INICIO_VALIDACION) & (df["fecha"] < C.FECHA_INICIO_TEST)]
    prueba = df[df["fecha"] >= C.FECHA_INICIO_TEST]
    return {"entrenamiento": entrenamiento, "validacion": validacion, "prueba": prueba, "todo": df}


# ==================================================================================================
# Modelos (pipelines de scikit-learn: el preprocesamiento se ajusta SOLO con los datos de entrenamiento)
# ==================================================================================================
def construir_modelo(nombre: str, numericas: list[str], n_estimadores: int | None = None) -> Pipeline:
    if nombre == "Regresión logística":
        prep = ColumnTransformer([
            ("num", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)), ("escalar", StandardScaler())]), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), VARS_CATEGORICAS),
        ])
        clf = LogisticRegression(max_iter=2000, C=0.5)
    elif nombre == "Random Forest":
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), numericas),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), VARS_CATEGORICAS),
        ])
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=60, max_features="sqrt", max_samples=0.5,
                                     n_jobs=-1, random_state=C.SEMILLA)
    elif nombre == "LightGBM":
        prep = ColumnTransformer([
            ("num", "passthrough", numericas),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), VARS_CATEGORICAS),
        ])
        clf = lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimadores or 3000, learning_rate=0.03,
                                 num_leaves=31, min_child_samples=300, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=2.0, random_state=C.SEMILLA, verbose=-1)
    else:
        raise ValueError(nombre)
    return Pipeline([("preprocesamiento", prep), ("modelo", clf)])


def probas_ordenadas(modelo: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """predict_proba con columnas en el orden Local, Empate, Visitante."""
    p = modelo.predict_proba(X)
    orden = [list(modelo.classes_).index(c) for c in CLASES]
    return p[:, orden]


def ajustar_lightgbm(numericas, X_tr, y_tr, X_va, y_va) -> tuple[Pipeline, int]:
    """LightGBM con parada temprana en validación; devuelve el modelo y el mejor número de árboles."""
    modelo = construir_modelo("LightGBM", numericas)
    prep = modelo.named_steps["preprocesamiento"].fit(X_tr)
    clf = modelo.named_steps["modelo"]
    cat_idx = [len(numericas)]
    # LightGBM 4.7 depreca eval_set en favor de eval_X/eval_y, pero la ruta nueva no codifica etiquetas de texto
    # ("Fields with bad pandas dtypes: resultado: str"); se mantiene eval_set y se silencia solo ese aviso.
    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
    clf.fit(prep.transform(X_tr), y_tr, eval_set=[(prep.transform(X_va), y_va)], eval_metric="multi_logloss",
            categorical_feature=cat_idx, callbacks=[lgb.early_stopping(100, verbose=False)])
    return modelo, int(clf.best_iteration_ or clf.n_estimators)


def entrenar(modelo: Pipeline, nombre: str, numericas, X, y) -> Pipeline:
    if nombre == "LightGBM":
        modelo.fit(X, y, modelo__categorical_feature=[len(numericas)])
    else:
        modelo.fit(X, y)
    return modelo


# ==================================================================================================
# Métricas
# ==================================================================================================
def metricas(y_real: pd.Series, probs: np.ndarray, pred: np.ndarray | None = None) -> dict:
    y = np.asarray(y_real)
    pred = np.array(CLASES)[probs.argmax(axis=1)] if pred is None else pred
    precision, recall, f1, soporte = precision_recall_fscore_support(y, pred, labels=CLASES, zero_division=0)
    onehot = np.stack([(y == c).astype(float) for c in CLASES], axis=1)
    probs_c = np.clip(probs, 1e-9, 1)
    probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
    # OJO (error encontrado y corregido): sklearn.metrics.log_loss asume que las columnas vienen en el orden
    # ALFABÉTICO de las etiquetas (Empate, Local, Visitante). Aquí van en orden Local, Empate, Visitante, así que
    # se calcula explícitamente: −promedio(log p_clase_real).
    log_loss_explicito = -np.mean(np.log(np.sum(probs_c * onehot, axis=1)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=CLASES, average="macro", zero_division=0)),
        "log_loss": float(log_loss_explicito),
        "brier": float(np.mean(np.sum((probs_c - onehot) ** 2, axis=1))),
        "roc_auc_ovr": float(roc_auc_score(onehot, probs_c, average="macro")),
        "por_clase": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "soporte": int(s)}
                      for c, p, r, f, s in zip(CLASES, precision, recall, f1, soporte)},
        "matriz_confusion": confusion_matrix(y, pred, labels=CLASES).tolist(),
    }


def predecir_con_umbral(probs: np.ndarray, umbral_empate: float) -> np.ndarray:
    """Predice Empate si p(empate) ≥ umbral; si no, el más probable entre Local y Visitante."""
    lv = np.where(probs[:, 0] >= probs[:, 2], "Local", "Visitante")
    return np.where(probs[:, 1] >= umbral_empate, "Empate", lv)


def buscar_umbral_empate(y: pd.Series, probs: np.ndarray) -> float:
    candidatos = np.round(np.arange(0.24, 0.40, 0.005), 3)
    puntajes = [f1_score(y, predecir_con_umbral(probs, u), labels=CLASES, average="macro") for u in candidatos]
    return float(candidatos[int(np.argmax(puntajes))])


def curva_calibracion(y: pd.Series, probs: np.ndarray, bins: int = 10) -> dict:
    salida = {}
    y = np.asarray(y)
    for i, c in enumerate(CLASES):
        p = probs[:, i]
        cortes = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, bins - 1)
        tabla = pd.DataFrame({"p": p, "real": (y == c).astype(float), "bin": idx}).groupby("bin").mean()
        salida[c] = {"prob_media": tabla["p"].round(4).tolist(), "frecuencia_real": tabla["real"].round(4).tolist()}
    return salida


# ==================================================================================================
# Orquestación
# ==================================================================================================
def ejecutar_modelado(partidos: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    datos = preparar_datos(partidos)
    tr, va, te = datos["entrenamiento"], datos["validacion"], datos["prueba"]
    trva = pd.concat([tr, va])
    y_tr, y_va, y_te, y_trva = tr["resultado"], va["resultado"], te["resultado"], trva["resultado"]
    log = print if verbose else (lambda *a, **k: None)
    log(f"Entrenamiento {len(tr):,} | validación {len(va):,} | prueba {len(te):,}")

    # ---------------- líneas base ----------------
    previas = y_tr.value_counts(normalize=True).reindex(CLASES).to_numpy()
    lineas_base = {}
    for nombre_split, d in [("validacion", va), ("prueba", te)]:
        n = len(d)
        prob_ingenua = np.tile(previas, (n, 1))
        lineas_base.setdefault("Siempre local", {})[nombre_split] = metricas(d["resultado"], prob_ingenua, np.array(["Local"] * n))
        prob_casa = d[["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"]].to_numpy(dtype="float64")
        lineas_base.setdefault("Favorito de la casa", {})[nombre_split] = metricas(d["resultado"], prob_casa)

    resultados, modelos_finales, importancias, calibracion = {}, {}, {}, {}
    predicciones = te[["fecha", "liga_codigo", "liga", "pais", "temporada", "local", "visitante", "resultado",
                       "goles_local", "goles_visitante", "cuota_local", "cuota_empate", "cuota_visitante",
                       "cuota_max_local", "cuota_max_empate", "cuota_max_visitante",
                       "prob_casa_local", "prob_casa_empate", "prob_casa_visitante"]].copy()
    for conjunto, numericas in CONJUNTOS.items():
        columnas = numericas + VARS_CATEGORICAS
        X_tr, X_va, X_te, X_trva = tr[columnas], va[columnas], te[columnas], trva[columnas]
        resultados[conjunto] = {}
        mejor_iter = None
        for nombre in ["Regresión logística", "Random Forest", "LightGBM"]:
            t = time.perf_counter()
            if nombre == "LightGBM":
                modelo, mejor_iter = ajustar_lightgbm(numericas, X_tr, y_tr, X_va, y_va)
            else:
                modelo = entrenar(construir_modelo(nombre, numericas), nombre, numericas, X_tr, y_tr)
            p_va = probas_ordenadas(modelo, X_va)          # modelo entrenado SOLO con entrenamiento
            m_va = metricas(y_va, p_va)
            # re-entrenamiento con entrenamiento + validación y evaluación final en prueba
            final = entrenar(construir_modelo(nombre, numericas, mejor_iter if nombre == "LightGBM" else None),
                             nombre, numericas, X_trva, y_trva)
            p_te = probas_ordenadas(final, X_te)
            m_te = metricas(y_te, p_te)
            resultados[conjunto][nombre] = {"validacion": m_va, "prueba": m_te, "segundos": round(time.perf_counter() - t, 1)}
            if nombre == "LightGBM":
                resultados[conjunto][nombre]["arboles"] = mejor_iter
            modelos_finales[(conjunto, nombre)] = (final, p_te, p_va)
            log(f"  [{conjunto:10s}] {nombre:20s} val logloss={m_va['log_loss']:.4f}  test logloss={m_te['log_loss']:.4f} "
                f"acc={m_te['accuracy']:.4f} f1m={m_te['f1_macro']:.4f}  ({resultados[conjunto][nombre]['segundos']} s)")

        # mejor modelo del conjunto por log-loss en validación
        mejor = min(resultados[conjunto], key=lambda k: resultados[conjunto][k]["validacion"]["log_loss"])
        final, p_te, p_va = modelos_finales[(conjunto, mejor)]
        # umbral de decisión para empates: se elige en validación con el modelo que NO vio validación
        umbral = buscar_umbral_empate(y_va, p_va)
        m_umbral = metricas(y_te, p_te, predecir_con_umbral(p_te, umbral))
        resultados[conjunto]["_mejor"] = {"modelo": mejor, "umbral_empate": umbral, "prueba_con_umbral": m_umbral}

        # importancia por permutación (muestra de prueba)
        muestra = te.sample(min(12000, len(te)), random_state=C.SEMILLA)
        imp = permutation_importance(final, muestra[columnas], muestra["resultado"], scoring="neg_log_loss",
                                     n_repeats=3, random_state=C.SEMILLA, n_jobs=1)
        importancias[conjunto] = sorted(
            [{"variable": c, "etiqueta": ETIQUETAS.get(c, c), "importancia": float(v), "desviacion": float(s)}
             for c, v, s in zip(columnas, imp.importances_mean, imp.importances_std)],
            key=lambda d: -d["importancia"])
        calibracion[conjunto] = curva_calibracion(y_te, p_te)

        for i, c in enumerate(CLASES):
            predicciones[f"p_{conjunto}_{c.lower()}"] = p_te[:, i]
        predicciones[f"pred_{conjunto}"] = np.array(CLASES)[p_te.argmax(axis=1)]

        joblib.dump({
            "pipeline": final, "nombre": mejor, "conjunto": conjunto, "numericas": numericas,
            "categoricas": VARS_CATEGORICAS, "clases": CLASES, "umbral_empate": umbral,
            "entrenado_hasta": str(trva["fecha"].max().date()), "filas_entrenamiento": int(len(trva)),
            "metricas_prueba": resultados[conjunto][mejor]["prueba"],
        }, C.MODELO_CON_CUOTAS if conjunto == "con_cuotas" else C.MODELO_SIN_CUOTAS, compress=3)

    calibracion["casa"] = curva_calibracion(y_te, te[["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"]].to_numpy())

    # ---------------- simulación de apuestas sobre prueba ----------------
    simulacion = {}
    # "casa" es la estrategia de control: usa las probabilidades de Bet365 contra la mejor cuota del mercado.
    # Si también gana, la ventaja viene de buscar la mejor cuota (line shopping), no del modelo.
    for conjunto in [*CONJUNTOS, "casa"]:
        cols = ([f"p_{conjunto}_{c.lower()}" for c in CLASES] if conjunto != "casa"
                else ["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"])
        _, resumen = simular_apuestas(predicciones, cols, umbral_valor=0.05)
        simulacion[conjunto] = {"umbral_5pct": resumen, "curva": curva_umbral(predicciones, cols).to_dict(orient="records")}

    predicciones.to_parquet(C.PARQUET_PREDICCIONES, index=False)

    # ---------------- criterios de éxito (definidos en Business Understanding) ----------------
    base_f1 = lineas_base["Siempre local"]["prueba"]["f1_macro"]
    casa = lineas_base["Favorito de la casa"]["prueba"]
    mejor_con = resultados["con_cuotas"]["_mejor"]
    mejor_sin = resultados["sin_cuotas"]["_mejor"]
    m_con = resultados["con_cuotas"][mejor_con["modelo"]]["prueba"]
    m_sin = resultados["sin_cuotas"][mejor_sin["modelo"]]["prueba"]
    criterios = {
        "f1_macro_vs_ingenua_pct": round(100 * (mejor_con["prueba_con_umbral"]["f1_macro"] / base_f1 - 1), 1),
        "cumple_f1_10pct": bool(mejor_con["prueba_con_umbral"]["f1_macro"] >= 1.10 * base_f1),
        "logloss_modelo_vs_casa": round(m_con["log_loss"] - casa["log_loss"], 5),
        "cumple_logloss_casa": bool(m_con["log_loss"] <= casa["log_loss"]),
        "accuracy_sin_cuotas_vs_casa_pp": round(100 * (m_sin["accuracy"] - casa["accuracy"]), 2),
        "roi_positivo": bool(simulacion["con_cuotas"]["umbral_5pct"]["roi"] > 0),
    }

    salida = {
        "tarea": "Clasificación multiclase (Local / Empate / Visitante)",
        "particion": {"entrenamiento": int(len(tr)), "validacion": int(len(va)), "prueba": int(len(te)),
                      "inicio_validacion": C.FECHA_INICIO_VALIDACION, "inicio_prueba": C.FECHA_INICIO_TEST,
                      "distribucion_prueba": y_te.value_counts(normalize=True).round(4).to_dict()},
        "variables": {k: v + VARS_CATEGORICAS for k, v in CONJUNTOS.items()},
        "lineas_base": lineas_base, "resultados": resultados, "importancias": importancias,
        "calibracion": calibracion, "simulacion": simulacion, "criterios_exito": criterios,
        "segundos_totales": round(time.perf_counter() - t0, 1),
    }
    C.JSON_MODELO.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Mejor con cuotas: {mejor_con['modelo']} | mejor sin cuotas: {mejor_sin['modelo']} | criterios: {criterios}")
    log(f"Modelado completo en {salida['segundos_totales']} s")
    return salida


if __name__ == "__main__":
    ejecutar_modelado()
