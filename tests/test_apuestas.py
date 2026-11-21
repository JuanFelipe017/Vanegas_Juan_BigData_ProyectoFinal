"""Prueba de la lógica de dinero de la simulación de apuestas. Ejecutar: python -m pytest tests  (o python tests/test_apuestas.py)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apuestas import simular_apuestas  # noqa: E402


def test_roi_beneficio_y_peor_racha():
    # 3 partidos: el modelo ve valor en el local siempre; gana 1 de 3 a cuota 3.0
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "liga_codigo": "E0", "local": ["A", "B", "C"], "visitante": ["X", "Y", "Z"],
        "resultado": ["Visitante", "Visitante", "Local"],
        "cuota_local": 3.0, "cuota_empate": 3.0, "cuota_visitante": 3.0,
        "cuota_max_local": 3.0, "cuota_max_empate": 3.0, "cuota_max_visitante": 3.0,
        "p_local": 0.5, "p_empate": 0.25, "p_visitante": 0.25,
    })
    apuestas, r = simular_apuestas(df, ["p_local", "p_empate", "p_visitante"], umbral_valor=0.05)
    assert r["apuestas"] == 3 and r["aciertos"] == 1
    assert r["beneficio_unidades"] == 0.0          # -1 -1 +2
    assert r["roi"] == 0.0
    assert r["max_caida"] == 2.0                    # banca 0 → -1 → -2: la caída cuenta desde 0
    assert list(apuestas["acumulado"]) == [-1.0, -2.0, 0.0]


if __name__ == "__main__":
    test_roi_beneficio_y_peor_racha()
    print("ok")
