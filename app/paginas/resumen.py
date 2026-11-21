"""Pestaña 1 — Resumen: KPIs del dataset y del modelo."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datos as D
import ui


def pagina() -> None:
    m_total = D.partidos()
    m = D.filtrar(m_total)
    etl = D.resumen_etl()
    met = D.metricas()
    doc, fuente = D.documento_mongo()

    ui.titulo("Proyecto final · Gerencia y almacenamiento en Big Data", "Fútbol & apuestas:", "¿se le gana a la casa?",
              f"Pipeline integral sobre {ui.num(etl['filas_crudas_total'], 0)} registros de tres fuentes públicas: resultados y cuotas "
              "(Football-Data.co.uk), ratings Elo (ClubElo) y valores de mercado (Transfermarkt).",
              [("Pipeline listo", ""), (f"Insights: {fuente.split(' (')[0]}", "oro" if fuente == "MongoDB" else "rojo")])

    ui.hero({
        "etiqueta": "EN VIVO DESDE EL PARQUET",
        "kicker": f"{etl['fecha_min'][:4]} – {etl['fecha_max'][:4]} · {etl['ligas']} ligas · {ui.num(etl['equipos'], 0)} equipos",
        "titulo": "El mercado sabe <span>casi todo</span>",
        "texto": "Un modelo con datos públicos se acerca a las casas de apuestas; con sus cuotas, las iguala. ¿Alcanza para ganar dinero?",
        "contadores": [(etl["partidos_limpios"], 0, "partidos limpios"), (etl["filas_crudas_total"], 0, "registros crudos"),
                       (etl["filas_formato_largo"], 0, "filas equipo × partido"), (met["particion"]["prueba"], 0, "partidos de prueba")],
    }, st.session_state["f_animaciones"])

    if doc:
        ui.ticker([(i["tema"], i["titulo"]) for i in doc["insights"]] * 2)

    # ---------------- KPIs del dataset (responden a los filtros del panel) ----------------
    st.caption(f"Filtros activos: {D.descripcion_filtros()}")
    if m.empty:
        st.warning("Los filtros no dejan partidos. Abra **Opciones** y amplíe la selección.")
        return
    pct = m["resultado"].value_counts(normalize=True) * 100
    pct_total = m_total["resultado"].value_counts(normalize=True) * 100
    margen = m["margen_casa"].mean() * 100
    acierto = m["acierto_favorito"].astype("float").mean() * 100

    def delta(actual, ref, sufijo=" pp", invertir=False):
        d = actual - ref
        clase = "neg" if (d < 0) != invertir else ""
        return f'<b class="{clase}">{"+" if d >= 0 else ""}{ui.num(d, 1)}{sufijo}</b> vs. total'

    ui.kpis([
        {"lbl": "Partidos", "val": ui.num(len(m), 0), "icono": "🏟️", "sub": f"{m['liga'].nunique()} ligas · {m['temporada'].nunique()} temporadas"},
        {"lbl": "Goles por partido", "val": ui.num(m["total_goles"].mean(), 2), "icono": "⚽", "acento": ui.ORO,
         "sub": delta(m["total_goles"].mean(), m_total["total_goles"].mean(), "")},
        {"lbl": "Gana el local", "val": ui.num(pct.get("Local", 0), 1), "unidad": "%", "icono": "🏠", "acento": ui.COLORES["Local"],
         "sub": delta(pct.get("Local", 0), pct_total["Local"])},
        {"lbl": "Empates", "val": ui.num(pct.get("Empate", 0), 1), "unidad": "%", "icono": "🤝", "acento": ui.COLORES["Empate"],
         "sub": delta(pct.get("Empate", 0), pct_total["Empate"])},
        {"lbl": "Margen de la casa", "val": ui.num(margen, 1), "unidad": "%", "icono": "🏦", "acento": ui.ROJO,
         "sub": "lo que se queda Bet365 por unidad"},
        {"lbl": "Acierta el favorito", "val": ui.num(acierto, 1), "unidad": "%", "icono": "🎯", "acento": ui.COLORES["Visitante"],
         "sub": f"{ui.num(m['sorpresa'].astype('float').mean() * 100, 1)} % de sorpresas (p < 25 %)"},
    ])

    # ---------------- marcador modelo vs casa ----------------
    mejor = met["resultados"]["con_cuotas"]["_mejor"]
    r_con = met["resultados"]["con_cuotas"][mejor["modelo"]]["prueba"]
    casa = met["lineas_base"]["Favorito de la casa"]["prueba"]
    sim = met["simulacion"]["con_cuotas"]["umbral_5pct"]
    ui.marcador("Modelo", f"{mejor['modelo']} · con cuotas", f"{ui.num(r_con['accuracy'] * 100, 1)} : {ui.num(casa['accuracy'] * 100, 1)}",
                "Casa", "Bet365 · favorito", f"Accuracy en {ui.num(met['particion']['prueba'], 0)} partidos de prueba · log-loss "
                f"<b>{ui.num(r_con['log_loss'], 4)}</b> vs. <b>{ui.num(casa['log_loss'], 4)}</b> · ROI apostando con valor ≥ 5 %: "
                f"<b style='color:var(--green)'>{ui.num(sim['roi'] * 100, 2)} %</b>")
    st.space("small")

    c1, c2 = st.columns([1, 1.6], gap="medium")
    with c1, st.container(key="tarjeta_donut"):
        st.markdown("### Resultado de los partidos")
        orden = [r for r in ui.COLORES if r in pct.index]
        fig = go.Figure(go.Pie(labels=orden, values=[pct[r] for r in orden], hole=0.62, sort=False,
                               marker=dict(colors=[ui.COLORES[r] for r in orden], line=dict(color="#050d09", width=3)),
                               textinfo="label+percent", textfont=dict(size=14, family="Barlow Condensed"),
                               hovertemplate="%{label}: %{value:.1f} %<extra></extra>"))
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                          annotations=[dict(text=f"<b>{ui.num(len(m), 0)}</b><br>partidos", showarrow=False,
                                            font=dict(size=18, family="Barlow Condensed", color=ui.TEXTO))])
        ui.grafico(fig, 330, key="donut")
    with c2, st.container(key="tarjeta_temporadas"):
        st.markdown("### Partidos por temporada y continente")
        t = m.groupby(["temporada_inicio", "continente"], observed=True).size().reset_index(name="partidos")
        fig = go.Figure()
        for cont, color in zip(["Europa", "América", "Asia"], ["#199e70", "#f4c542", "#3987e5"]):
            s = t[t["continente"] == cont]
            if len(s):
                fig.add_bar(x=s["temporada_inicio"], y=s["partidos"], name=cont, marker_color=color,
                            hovertemplate=f"{cont} · %{{x}}: %{{y:,}} partidos<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.18, xaxis_title="Temporada (año de inicio)", yaxis_title="Partidos")
        ui.grafico(fig, 330, key="temporadas")

    # ---------------- KPIs del modelo ----------------
    st.markdown("### Indicadores del modelo")
    sin = met["resultados"]["sin_cuotas"]["_mejor"]
    r_sin = met["resultados"]["sin_cuotas"][sin["modelo"]]["prueba"]
    ingenua = met["lineas_base"]["Siempre local"]["prueba"]
    ui.kpis([
        {"lbl": "Accuracy con cuotas", "val": ui.num(r_con["accuracy"] * 100, 1), "unidad": "%", "icono": "🤖",
         "sub": f'<b>+{ui.num((r_con["accuracy"] - ingenua["accuracy"]) * 100, 1)} pp</b> vs. "siempre local"'},
        {"lbl": "Accuracy fútbol puro", "val": ui.num(r_sin["accuracy"] * 100, 1), "unidad": "%", "icono": "📐", "acento": ui.ORO,
         "sub": f"{sin['modelo']} · sin mirar cuotas"},
        {"lbl": "F1 macro (umbral empate)", "val": ui.num(mejor["prueba_con_umbral"]["f1_macro"], 3), "icono": "⚖️",
         "acento": ui.COLORES["Empate"], "sub": f'<b>+{ui.num(met["criterios_exito"]["f1_macro_vs_ingenua_pct"], 0)} %</b> sobre la línea base'},
        {"lbl": "Log-loss", "val": ui.num(r_con["log_loss"], 4), "icono": "📉", "acento": ui.COLORES["Visitante"],
         "sub": f"casa {ui.num(casa['log_loss'], 4)} · ingenua {ui.num(ingenua['log_loss'], 4)}"},
        {"lbl": "ROI simulado", "val": ui.num(sim["roi"] * 100, 2), "unidad": "%", "icono": "💰", "acento": ui.VERDE if sim["roi"] > 0 else ui.ROJO,
         "sub": f"{ui.num(sim['apuestas'], 0)} apuestas · {ui.num(sim['beneficio_unidades'], 0)} u"},
    ])
    with st.expander("¿Cómo leer estos indicadores?"):
        st.markdown(
            "- **Accuracy**: % de partidos donde el resultado más probable según el modelo fue el real. Adivinar siempre "
            f"*local* da {ui.num(ingenua['accuracy'] * 100, 1)} %.\n"
            "- **Log-loss**: castiga la confianza equivocada; es la métrica principal porque evalúa *probabilidades*, que es lo "
            "que importa para apostar. Menor es mejor.\n"
            "- **F1 macro**: promedia el F1 de las tres clases, así que obliga a acertar también empates y visitantes.\n"
            "- **ROI**: beneficio por unidad apostada en la simulación con la mejor cuota del mercado (pestaña *Modelo → Apuestas*).")
