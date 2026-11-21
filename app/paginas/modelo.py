"""Pestaña 4 — Modelo: métricas, comparación, interpretación, simulador de partido y simulador de apuestas."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datos as D
import ui
from apuestas import simular_apuestas

ORDEN = ["Local", "Empate", "Visitante"]
NOMBRES = {"con_cuotas": "Con cuotas", "sin_cuotas": "Sin cuotas (fútbol puro)"}


def pagina() -> None:
    met = D.metricas()
    ui.titulo("CRISP-DM · Modeling & Evaluation", "El modelo contra", "la casa",
              f"Clasificación Local / Empate / Visitante con partición temporal: se entrena hasta junio de 2023 y se evalúa en "
              f"{ui.num(met['particion']['prueba'], 0)} partidos de 2023/24 en adelante que el modelo nunca vio.",
              [("Modelo cargado desde models/ (sin re-entrenar)", "")])
    t1, t2, t3, t4 = st.tabs(["🏆 Rendimiento", "🧠 Interpretación", "🔮 Simulador de partido", "💰 Simulador de apuestas"])
    with t1:
        rendimiento(met)
    with t2:
        interpretacion(met)
    with t3:
        simulador_partido(met)
    with t4:
        simulador_apuestas(met)


# ==================================================================================================
def rendimiento(met: dict) -> None:
    filas = []
    for nombre, r in met["lineas_base"].items():
        filas.append({"Modelo": nombre, "Tipo": "Línea base", **{k: r["prueba"][k] for k in ["accuracy", "f1_macro", "log_loss", "brier", "roc_auc_ovr"]}})
    for conj, res in met["resultados"].items():
        for nombre, r in res.items():
            if not nombre.startswith("_"):
                filas.append({"Modelo": f"{nombre} · {NOMBRES[conj].lower()}", "Tipo": NOMBRES[conj],
                              **{k: r["prueba"][k] for k in ["accuracy", "f1_macro", "log_loss", "brier", "roc_auc_ovr"]}})
    tabla = pd.DataFrame(filas)
    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1, st.container(key="tarjeta_m1"):
        metrica = st.segmented_control("Métrica", ["log_loss", "accuracy", "f1_macro", "brier"], default="log_loss", key="m_metrica",
                                       format_func={"log_loss": "Log-loss", "accuracy": "Accuracy", "f1_macro": "F1 macro", "brier": "Brier"}.get)
        metrica = metrica or "log_loss"
        menor_mejor = metrica in ("log_loss", "brier")
        t = tabla.sort_values(metrica, ascending=not menor_mejor)
        colores = {"Línea base": "#5b6b62", "Con cuotas": ui.VERDE, "Sin cuotas (fútbol puro)": ui.ORO}
        fig = go.Figure(go.Bar(x=t[metrica], y=t["Modelo"], orientation="h", marker_color=[colores[x] for x in t["Tipo"]],
                               text=[f"{v:.4f}" for v in t[metrica]], textposition="outside", textfont=dict(color=ui.MUTED),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        casa = met["lineas_base"]["Favorito de la casa"]["prueba"][metrica]
        fig.add_vline(x=casa, line_dash="dash", line_color=ui.TEXTO, annotation_text="casa", annotation_font_color=ui.MUTED)
        rango = [t[metrica].min() * (0.985 if menor_mejor else 0.9), t[metrica].max() * (1.01 if menor_mejor else 1.08)]
        fig.update_layout(title=f"Comparación en prueba ({'menor' if menor_mejor else 'mayor'} es mejor)", xaxis=dict(range=rango),
                          yaxis=dict(autorange="reversed"))
        ui.grafico(fig, 430, key="m_comp")
    with c2, st.container(key="tarjeta_m2"):
        conj = st.segmented_control("Conjunto", list(NOMBRES), default="con_cuotas", format_func=NOMBRES.get, key="m_conj_cm") or "con_cuotas"
        modo = st.toggle("Usar umbral de empate (optimizado en validación)", value=True, key="m_umbral")
        mejor = met["resultados"][conj]["_mejor"]
        r = mejor["prueba_con_umbral"] if modo else met["resultados"][conj][mejor["modelo"]]["prueba"]
        mc = np.array(r["matriz_confusion"])
        pct = mc / mc.sum(axis=1, keepdims=True) * 100
        fig = go.Figure(go.Heatmap(z=pct, x=ORDEN, y=ORDEN, colorscale=ui.SECUENCIAL, zmin=0, zmax=80, showscale=False,
                                   text=[[f"{pct[i, j]:.0f}%<br>({mc[i, j]:,})" for j in range(3)] for i in range(3)],
                                   texttemplate="%{text}", textfont=dict(size=14, family="Barlow Condensed"),
                                   hovertemplate="Real %{y} → predicho %{x}: %{z:.1f} %<extra></extra>"))
        fig.update_layout(title=f"Matriz de confusión · {mejor['modelo']}", xaxis_title="Predicción", yaxis=dict(title="Real", autorange="reversed"))
        ui.grafico(fig, 360, key="m_cm")
        st.caption(f"Umbral: predecir empate si p(empate) ≥ {mejor['umbral_empate']:.3f}. Sube el recall de empates a costa de algo de accuracy.")

    st.dataframe(tabla, hide_index=True, column_config={
        "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.4, max_value=0.55, format="%.4f"),
        "f1_macro": st.column_config.NumberColumn("F1 macro", format="%.4f"), "log_loss": st.column_config.NumberColumn("Log-loss", format="%.4f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.4f"), "roc_auc_ovr": st.column_config.NumberColumn("ROC-AUC", format="%.4f")})
    ce = met["criterios_exito"]
    ui.kpis([
        {"lbl": "Criterio 1 · F1 macro ≥ +10 % vs. ingenua", "val": "✔ Cumple" if ce["cumple_f1_10pct"] else "✘ No cumple",
         "sub": f"<b>+{ui.num(ce['f1_macro_vs_ingenua_pct'], 1)} %</b>", "acento": ui.VERDE if ce["cumple_f1_10pct"] else ui.ROJO},
        {"lbl": "Criterio 2 · log-loss ≤ casa", "val": "✔ Cumple" if ce["cumple_logloss_casa"] else "✘ No cumple",
         "sub": f"diferencia {ui.num(ce['logloss_modelo_vs_casa'], 5)}", "acento": ui.VERDE if ce["cumple_logloss_casa"] else ui.ROJO},
        {"lbl": "Extra · ROI positivo", "val": "✔ Sí" if ce["roi_positivo"] else "✘ No", "acento": ui.ORO,
         "sub": "simulación con valor ≥ 5 % y mejores cuotas"},
        {"lbl": "Fútbol puro vs. casa", "val": f"{ui.num(ce['accuracy_sin_cuotas_vs_casa_pp'], 2)} pp", "acento": ui.COLORES["Visitante"],
         "sub": "accuracy sin usar cuotas"},
    ])


# ==================================================================================================
def interpretacion(met: dict) -> None:
    conj = st.segmented_control("Modelo", list(NOMBRES), default="sin_cuotas", format_func=NOMBRES.get, key="m_conj_imp") or "sin_cuotas"
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1, st.container(key="tarjeta_m3"):
        imp = pd.DataFrame(met["importancias"][conj]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importancia"], y=imp["etiqueta"], orientation="h",
                               error_x=dict(type="data", array=imp["desviacion"], color=ui.MUTED, thickness=1),
                               marker=dict(color=imp["importancia"], colorscale=ui.SECUENCIAL),
                               hovertemplate="%{y}: %{x:.5f}<extra></extra>"))
        fig.update_layout(title="Importancia por permutación (Δ log-loss al desordenar)", xaxis_title="Importancia")
        ui.grafico(fig, 470, key="m_imp")
    with c2, st.container(key="tarjeta_m4"):
        cal = met["calibracion"]
        clase = st.pills("Clase", ORDEN, default="Local", key="m_cal_clase") or "Local"
        fig = go.Figure()
        fig.add_scatter(x=[0, 0.85], y=[0, 0.85], mode="lines", line=dict(color=ui.MUTED, dash="dash", width=1), showlegend=False)
        for fuente, nombre, color in [(conj, "Modelo", ui.COLORES[clase]), ("casa", "Casa (Bet365)", ui.MUTED)]:
            d = cal[fuente][clase]
            fig.add_scatter(x=d["prob_media"], y=d["frecuencia_real"], mode="lines+markers", name=nombre, line=dict(color=color, width=3),
                            hovertemplate=f"{nombre}: predice %{{x:.0%}} → ocurre %{{y:.0%}}<extra></extra>")
        fig.update_layout(title=f"Calibración en prueba · {clase}", xaxis=dict(title="Probabilidad predicha", tickformat=".0%"),
                          yaxis=dict(title="Frecuencia real", tickformat=".0%"))
        ui.grafico(fig, 470, key="m_cal")
    top = pd.DataFrame(met["importancias"][conj])
    st.markdown("#### Lectura en lenguaje de negocio")
    if conj == "sin_cuotas":
        st.markdown(
            f"- **{top.iloc[0]['etiqueta']}** domina: saber cuánto mejor es un equipo que otro (historial completo resumido en un número) "
            "explica más que cualquier racha reciente.\n"
            f"- Después vienen **{top.iloc[1]['etiqueta']}** y **{top.iloc[2]['etiqueta']}**: la liga cambia la tasa de empates y de "
            "victorias locales, y el valor de plantilla aporta calidad individual que el Elo tarda en reflejar (fichajes).\n"
            "- La **forma de los últimos 5 partidos** pesa muy poco una vez que se conoce el Elo: las rachas se sobrevaloran.")
    else:
        st.markdown(
            f"- **{top.iloc[0]['etiqueta']}** concentra casi toda la información: la cuota ya resume Elo, forma, lesiones y noticias.\n"
            "- Las variables futbolísticas añaden poco encima del mercado, pero lo suficiente para encontrar algunas cuotas con valor "
            "(ver *Simulador de apuestas*).\n"
            "- El modelo está tan bien calibrado como la casa: cuando dice 50 %, ocurre ~50 % de las veces.")


# ==================================================================================================
def simulador_partido(met: dict) -> None:
    estado = D.estado_equipos()
    modelos = D.modelos()
    recientes = estado[estado["ultima_fecha"] >= estado["ultima_fecha"].max() - pd.Timedelta(days=420)]
    ligas = recientes.groupby("liga")["equipo"].count()
    ligas = ligas[ligas >= 8].index.tolist()
    st.markdown("Elija dos equipos de la misma liga. El modelo usa su **último estado conocido** (Elo propio, forma, puntos por "
                "partido, récord como local/visitante y valor de plantilla) y predice sin re-entrenar.")
    c0, c1, c2 = st.columns([1.2, 1, 1])
    liga = c0.selectbox("Liga", sorted(ligas), index=sorted(ligas).index("Premier League") if "Premier League" in ligas else 0)
    equipos = recientes[recientes["liga"] == liga].sort_values("elo_propio", ascending=False)
    nombres = equipos["equipo"].tolist()
    local = c1.selectbox("🏠 Local", nombres, index=0)
    visitante = c2.selectbox("🚗 Visitante", [n for n in nombres if n != local], index=0)
    c3, c4, c5 = st.columns(3)
    descanso_l = c3.slider("Días de descanso del local", 2, 14, 7)
    descanso_v = c4.slider("Días de descanso del visitante", 2, 14, 7)
    mes = c5.select_slider("Mes del partido", list(range(1, 13)), value=date.today().month,
                           format_func=lambda x: ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"][x - 1])
    usar_cuotas = st.toggle("Tengo las cuotas de una casa (usa el modelo con cuotas)", value=False)
    cuotas = None
    if usar_cuotas:
        q1, q2, q3 = st.columns(3)
        cuotas = (q1.number_input("Cuota local (1)", 1.01, 50.0, 2.10, 0.05), q2.number_input("Cuota empate (X)", 1.01, 50.0, 3.40, 0.05),
                  q3.number_input("Cuota visitante (2)", 1.01, 50.0, 3.60, 0.05))

    fila_l, fila_v = equipos.set_index("equipo").loc[local], equipos.set_index("equipo").loc[visitante]
    nivel = int(fila_l["nivel"])
    codigo = str(fila_l["liga_codigo"])
    fila = D.fila_prediccion(fila_l, fila_v, codigo, nivel, mes, descanso_l, descanso_v, cuotas)
    paquete = modelos["con_cuotas" if usar_cuotas else "sin_cuotas"]
    probs = D.predecir(paquete, fila)

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, st.container(key="tarjeta_m5"):
        st.markdown(f"### {local} <span style='color:var(--muted)'>vs.</span> {visitante}", unsafe_allow_html=True)
        st.caption(f"Modelo: {paquete['nombre']} · {NOMBRES[paquete['conjunto']]} · entrenado hasta {paquete['entrenado_hasta']}")
        st.html(ui.barras_probabilidad(probs))
        justas = {k: 1 / v for k, v in probs.items()}
        celdas = []
        for i, k in enumerate(ORDEN):
            valor = cuotas is not None and probs[k] * cuotas[i] > 1.0
            extra = f"<div class='k' style='margin-top:.3rem'>casa {cuotas[i]:.2f} · VE {probs[k] * cuotas[i] - 1:+.1%}</div>" if cuotas else ""
            celdas.append(f"<div class='sb-cuota {'valor' if valor else ''}'><div class='k'>Cuota justa {k}</div>"
                          f"<div class='v'>{justas[k]:.2f}</div>{extra}</div>")
        st.html(f"<div class='sb-cuotas'>{''.join(celdas)}</div>")
        if cuotas:
            mejores = [k for i, k in enumerate(ORDEN) if probs[k] * cuotas[i] > 1.05]
            if mejores:
                st.success(f"Apuesta con valor según el modelo: **{', '.join(mejores)}** (valor esperado > 5 %).", icon="💰")
            else:
                st.info("Ninguna cuota tiene valor esperado > 5 % según el modelo: mejor no apostar.", icon="🛑")
    with c2, st.container(key="tarjeta_m6"):
        comp = pd.DataFrame({
            "Variable": ["Elo propio", "Elo ClubElo", "Puntos prom. (5)", "Goles a favor (5)", "Goles en contra (5)", "Pts/partido temporada",
                         "Récord local / visitante (5)", "Valor plantilla (M€)"],
            local: [fila_l["elo_propio"], fila_l["elo_clubelo"], fila_l["puntos_prom5"], fila_l["gf_prom5"], fila_l["gc_prom5"],
                    fila_l["ppg_temporada"], fila_l["record_local"], fila_l["valor_plantilla"]],
            visitante: [fila_v["elo_propio"], fila_v["elo_clubelo"], fila_v["puntos_prom5"], fila_v["gf_prom5"], fila_v["gc_prom5"],
                        fila_v["ppg_temporada"], fila_v["record_visitante"], fila_v["valor_plantilla"]],
        })
        st.markdown("### Cara a cara")
        st.dataframe(comp, hide_index=True, column_config={local: st.column_config.NumberColumn(format="%.2f"),
                                                          visitante: st.column_config.NumberColumn(format="%.2f")})
        ep = D.equipos_partido()
        h2h = ep[(ep["equipo"] == local) & (ep["rival"] == visitante)].sort_values("fecha", ascending=False).head(8)
        if len(h2h):
            st.caption("Últimos enfrentamientos (desde el punto de vista del local elegido)")
            st.dataframe(h2h.assign(marcador=h2h["gf"].astype(int).astype(str) + " – " + h2h["gc"].astype(int).astype(str),
                                    condicion=np.where(h2h["es_local"], "Local", "Visitante"))[["fecha", "condicion", "marcador"]],
                         hide_index=True, column_config={"fecha": st.column_config.DateColumn(format="DD/MM/YYYY")})
        st.caption(f"Último partido registrado: {local} {fila_l['ultima_fecha']:%d/%m/%Y} · {visitante} {fila_v['ultima_fecha']:%d/%m/%Y}")


# ==================================================================================================
def simulador_apuestas(met: dict) -> None:
    pred = D.predicciones()
    s = st.session_state
    if s["f_paises"]:
        pred = pred[pred["pais"].isin(s["f_paises"])]
    if s["f_ligas"]:
        pred = pred[pred["liga"].isin(s["f_ligas"])]
    st.markdown("Estrategia de **apuestas de valor**: se apuesta 1 unidad solo cuando `probabilidad × cuota − 1` supera el umbral. "
                "Se evalúa en las temporadas de prueba (respeta los filtros de país y liga del panel).")
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1])
    estrategia = c1.segmented_control("Probabilidades", ["con_cuotas", "sin_cuotas", "casa"], default="con_cuotas", key="ap_estr",
                                      format_func={"con_cuotas": "Modelo con cuotas", "sin_cuotas": "Modelo sin cuotas", "casa": "Control: casa"}.get) or "con_cuotas"
    umbral = c2.slider("Umbral de valor esperado", 0.0, 0.30, 0.05, 0.01, format="%.2f")
    cuota_max = c3.slider("Cuota máxima", 2.0, 15.0, 10.0, 0.5)
    maximas = c4.toggle("Mejor cuota del mercado", value=True, help="Si se apaga, se usan solo las cuotas de Bet365")
    cols = ([f"p_{estrategia}_{c.lower()}" for c in ORDEN] if estrategia != "casa" else ["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"])
    apuestas, res = simular_apuestas(pred, cols, umbral, maximas, cuota_max)
    if res["apuestas"] == 0:
        st.warning("Con esos parámetros no se hace ninguna apuesta.")
        return
    ui.kpis([
        {"lbl": "Apuestas", "val": ui.num(res["apuestas"], 0), "icono": "🎟️", "sub": f"de {ui.num(res['partidos_evaluados'], 0)} partidos"},
        {"lbl": "Aciertos", "val": ui.num(res["tasa_acierto"] * 100, 1), "unidad": "%", "icono": "✅", "acento": ui.COLORES["Visitante"],
         "sub": f"cuota media {ui.num(res['cuota_media'], 2)}"},
        {"lbl": "Beneficio", "val": f"{res['beneficio_unidades']:+,.1f}".replace(",", "."), "unidad": "u", "icono": "💵",
         "acento": ui.VERDE if res["beneficio_unidades"] > 0 else ui.ROJO, "sub": "1 unidad por apuesta"},
        {"lbl": "ROI", "val": f"{res['roi'] * 100:+.2f}".replace(".", ","), "unidad": "%", "icono": "📈",
         "acento": ui.VERDE if res["roi"] > 0 else ui.ROJO, "sub": "beneficio / unidades apostadas"},
        {"lbl": "Peor racha", "val": ui.num(res["max_caida"], 1), "unidad": "u", "icono": "📉", "acento": ui.ROJO, "sub": "caída máxima desde un pico"},
    ])
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, st.container(key="tarjeta_m7"):
        fig = go.Figure()
        color = ui.VERDE if res["beneficio_unidades"] >= 0 else ui.ROJO
        fig.add_scatter(x=apuestas["fecha"], y=apuestas["acumulado"], mode="lines", line=dict(color=color, width=2.5), fill="tozeroy",
                        fillcolor="rgba(34,224,138,.08)" if res["beneficio_unidades"] >= 0 else "rgba(230,103,103,.08)",
                        hovertemplate="%{x|%d %b %Y}: %{y:+.1f} u<extra></extra>", name="Banca")
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="Evolución de la banca (unidades)", yaxis_title="Beneficio acumulado", showlegend=False)
        ui.grafico(fig, 400, key="ap_banca")
    with c2, st.container(key="tarjeta_m8"):
        por = apuestas.groupby("apuesta").agg(n=("ganancia", "size"), roi=("ganancia", "mean")).reindex(ORDEN).dropna()
        fig = go.Figure(go.Bar(x=por.index, y=por["roi"] * 100, marker_color=[ui.COLORES[k] for k in por.index],
                               text=[f"{v * 100:+.1f}%<br>{int(n)} ap." for v, n in zip(por["roi"], por["n"])], textposition="outside",
                               hovertemplate="%{x}: ROI %{y:.2f} %<extra></extra>"))
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="ROI por tipo de apuesta", yaxis_title="ROI (%)")
        ui.grafico(fig, 400, key="ap_tipo")
    curva = pd.DataFrame([simular_apuestas(pred, cols, u, maximas, cuota_max)[1] for u in np.round(np.arange(0, 0.31, 0.02), 2)])
    curva = curva[curva["apuestas"] >= 30]
    with st.container(key="tarjeta_m9"):
        fig = go.Figure()
        fig.add_bar(x=curva["umbral_valor"] * 100, y=curva["apuestas"], name="Apuestas", marker_color="rgba(57,135,229,.35)", yaxis="y2",
                    hovertemplate="Umbral %{x:.0f} %: %{y:,} apuestas<extra></extra>")
        fig.add_scatter(x=curva["umbral_valor"] * 100, y=curva["roi"] * 100, name="ROI", mode="lines+markers", line=dict(color=ui.ORO, width=3),
                        hovertemplate="Umbral %{x:.0f} %: ROI %{y:+.2f} %<extra></extra>")
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="Más exigencia = menos apuestas (y más varianza)", xaxis_title="Umbral de valor esperado (%)",
                          yaxis=dict(title="ROI (%)"), yaxis2=dict(overlaying="y", side="right", showgrid=False, visible=False))
        ui.grafico(fig, 360, key="ap_curva")
        st.caption("Las barras (número de apuestas, sin eje) dan contexto: con pocas apuestas el ROI es pura varianza.")
    st.markdown("#### Últimas apuestas de la simulación")
    st.dataframe(apuestas.sort_values("fecha", ascending=False).head(60), hide_index=True, column_config={
        "fecha": st.column_config.DateColumn(format="DD/MM/YYYY"), "cuota": st.column_config.NumberColumn(format="%.2f"),
        "prob_modelo": st.column_config.ProgressColumn("Prob.", min_value=0, max_value=1, format="percent"),
        "valor_esperado": st.column_config.NumberColumn("VE", format="%.1%"), "ganancia": st.column_config.NumberColumn(format="%+.2f"),
        "acumulado": st.column_config.NumberColumn(format="%+.1f")})
    st.warning("Simulación académica con datos históricos: no es una recomendación de apuesta. Las casas limitan a quien gana "
               "seguido y las cuotas máximas no siempre están disponibles al mismo tiempo.", icon="⚠️")
