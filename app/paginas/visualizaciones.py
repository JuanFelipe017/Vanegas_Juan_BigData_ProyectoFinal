"""Pestaña 3 — Visualizaciones interactivas (todas responden a los filtros del panel flotante)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import datos as D
import ui

ORDEN = ["Local", "Empate", "Visitante"]


def pagina() -> None:
    m = D.filtrar(D.partidos())
    ui.titulo("Análisis exploratorio interactivo", "Lo que dicen", "los datos",
              "Cada gráfica responde una pregunta de negocio. Pase el mouse para ver el detalle, haga zoom arrastrando y use "
              "<b>Opciones</b> para filtrar temporadas, países y ligas.", [(D.descripcion_filtros(), "oro")])
    if m.empty:
        st.warning("Los filtros no dejan partidos.")
        return
    t1, t2, t3, t4, t5 = st.tabs(["📈 Tendencias", "🌍 Ligas y países", "🎲 Mercado de apuestas", "🛡️ Equipos", "🎬 Animadas"])

    # ================================================================== tendencias
    with t1:
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="tarjeta_v1"):
            pct = pd.crosstab(m["temporada_inicio"], m["resultado"], normalize="index")[ORDEN] * 100
            fig = go.Figure()
            for r in ORDEN:
                fig.add_scatter(x=pct.index, y=pct[r], name=r, mode="lines+markers", line=dict(color=ui.COLORES[r], width=3),
                                marker=dict(size=7), hovertemplate=f"{r} · %{{x}}: %{{y:.1f}} %<extra></extra>")
            if 2020 in pct.index:
                fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="rgba(255,255,255,.06)", line_width=0,
                              annotation_text="Estadios vacíos", annotation_position="top left", annotation_font_color=ui.MUTED)
            fig.update_layout(title=f"Ventaja local: {ui.num(pct['Local'].iloc[0], 0)} % → {ui.num(pct['Local'].iloc[-1], 0)} %",
                              yaxis_title="% de partidos", xaxis_title="Temporada", hovermode="x unified")
            ui.grafico(fig, 400, key="v_tend1")
        with c2, st.container(key="tarjeta_v2"):
            g = m.groupby("temporada_inicio").agg(goles=("total_goles", "mean"), mas25=("mas_de_2_5", "mean"), btts=("ambos_marcan", "mean"))
            fig = go.Figure()
            fig.add_scatter(x=g.index, y=g["goles"], mode="lines+markers", line=dict(color=ui.ORO, width=3), fill="tozeroy",
                            fillcolor="rgba(244,197,66,.08)", name="Goles por partido", hovertemplate="%{x}: %{y:.2f} goles<extra></extra>")
            fig.update_layout(title=f"Goles por partido (máx. {ui.num(g['goles'].max(), 2)} en {g['goles'].idxmax()})",
                              yaxis=dict(title="Goles promedio", range=[g["goles"].min() * 0.9, g["goles"].max() * 1.05]), xaxis_title="Temporada")
            ui.grafico(fig, 400, key="v_tend2")
        with st.container(key="tarjeta_v3"):
            from math import exp, factorial
            tot = m["total_goles"].astype(int).clip(upper=9)
            frec = tot.value_counts(normalize=True).sort_index() * 100
            lam = m["total_goles"].mean()
            fig = go.Figure()
            fig.add_bar(x=frec.index, y=frec.values, name="Observado", marker_color=ui.COLORES["Visitante"],
                        hovertemplate="%{x} goles: %{y:.1f} %<extra></extra>")
            fig.add_scatter(x=frec.index, y=[exp(-lam) * lam ** k / factorial(k) * 100 for k in frec.index], name=f"Poisson λ={lam:.2f}",
                            mode="lines+markers", line=dict(color=ui.TEXTO, width=2, dash="dot"))
            fig.update_layout(title="Distribución de goles por partido vs. Poisson", xaxis_title="Goles totales (9 = 9 o más)",
                              yaxis_title="% de partidos", bargap=0.25)
            ui.grafico(fig, 360, key="v_tend3")

    # ================================================================== ligas
    with t2:
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1, st.container(key="tarjeta_v4"):
            g = m.groupby(["continente", "pais", "liga"], observed=True).agg(partidos=("total_goles", "size"),
                                                                                goles=("total_goles", "mean")).reset_index()
            fig = px.sunburst(g, path=["continente", "pais", "liga"], values="partidos", color="goles",
                              color_continuous_scale=ui.SECUENCIAL, hover_data={"goles": ":.2f"})
            fig.update_traces(marker=dict(line=dict(color="#050d09", width=1.5)), insidetextfont=dict(family="Barlow Condensed", size=13))
            fig.update_layout(title="Drill-down continente → país → liga (color = goles/partido)", coloraxis_colorbar=dict(title="Goles"))
            ui.grafico(fig, 520, key="v_sun")
        with c2, st.container(key="tarjeta_v5"):
            g = m.groupby("liga", observed=True).agg(goles=("total_goles", "mean"), n=("total_goles", "size"))
            g = g[g["n"] >= 200].sort_values("goles")
            fig = go.Figure(go.Bar(x=g["goles"], y=g.index, orientation="h", marker=dict(color=g["goles"], colorscale=ui.SECUENCIAL),
                                   text=[f"{v:.2f}" for v in g["goles"]], textposition="outside", textfont=dict(color=ui.MUTED),
                                   hovertemplate="%{y}: %{x:.2f} goles · %{customdata:,} partidos<extra></extra>", customdata=g["n"]))
            fig.update_layout(title="Goles por partido por liga", xaxis=dict(title="Goles promedio", range=[g["goles"].min() * 0.85, g["goles"].max() * 1.08]))
            ui.grafico(fig, 520, key="v_ligas")
        with st.container(key="tarjeta_v6"):
            primera = m[m["nivel"] == 1]
            if primera.empty:
                primera = m
            piv = primera.pivot_table(index="pais", columns="temporada_inicio", values="resultado",
                                      aggfunc=lambda s: (s == "Local").mean() * 100, observed=True)
            piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
            fig = px.imshow(piv, color_continuous_scale=ui.SECUENCIAL, aspect="auto", zmin=30, zmax=60,
                            labels=dict(x="Temporada", y="País", color="% local"))
            fig.update_traces(hovertemplate="%{y} · %{x}: %{z:.1f} % de victorias locales<extra></extra>")
            fig.update_layout(title="Cubo país × temporada: % de victorias locales (primeras divisiones)")
            ui.grafico(fig, 200 + 20 * len(piv), key="v_heat")

    # ================================================================== mercado
    with t3:
        d = m.dropna(subset=["prob_casa_local"])
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="tarjeta_v7"):
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, dash="dash", width=1), showlegend=False, hoverinfo="skip")
            for r in ORDEN:
                p = d[f"prob_casa_{r.lower()}"]
                bins = pd.cut(p, np.linspace(0, 1, 26))
                t = pd.DataFrame({"p": p, "real": d["resultado"] == r}).groupby(bins, observed=True).agg(p=("p", "mean"), real=("real", "mean"), n=("p", "size"))
                t = t[t["n"] >= 60]
                fig.add_scatter(x=t["p"], y=t["real"], mode="lines+markers", name=r, line=dict(color=ui.COLORES[r], width=2.5),
                                marker=dict(size=np.clip(np.sqrt(t["n"]) / 3, 6, 18)), customdata=t["n"],
                                hovertemplate=f"{r}<br>Casa dice %{{x:.0%}} → pasa %{{y:.0%}}<br>%{{customdata:,}} partidos<extra></extra>")
            fig.update_layout(title="¿Están calibradas las casas? (casi perfecto)", xaxis=dict(title="Probabilidad implícita", tickformat=".0%"),
                              yaxis=dict(title="Frecuencia real", tickformat=".0%"))
            ui.grafico(fig, 440, key="v_calib")
        with c2, st.container(key="tarjeta_v8"):
            g = d[d["temporada_inicio"] <= 2025].groupby(["temporada_inicio", "continente"], observed=True)["margen_casa"].mean().mul(100).reset_index()
            fig = px.line(g, x="temporada_inicio", y="margen_casa", color="continente", markers=True,
                          color_discrete_map={"Europa": "#199e70", "América": "#f4c542", "Asia": "#3987e5"})
            fig.update_traces(line_width=3, hovertemplate="%{x}: %{y:.2f} %<extra>%{fullData.name}</extra>")
            fig.update_layout(title="El margen de la casa no para de bajar", xaxis_title="Temporada", yaxis_title="Margen promedio (%)")
            ui.grafico(fig, 440, key="v_margen")
        c1, c2 = st.columns([1, 1.2], gap="medium")
        with c1, st.container(key="tarjeta_v9"):
            g = d.groupby("liga", observed=True).agg(acierto=("acierto_favorito", "mean"), n=("liga", "size"))
            g = g[g["n"] >= 300].sort_values("acierto")
            g["acierto"] = g["acierto"].astype(float) * 100
            fig = go.Figure(go.Bar(x=g["acierto"], y=g.index, orientation="h", marker_color=ui.COLORES["Visitante"],
                                   hovertemplate="%{y}: el favorito gana %{x:.1f} %<extra></extra>"))
            fig.update_layout(title="¿Dónde gana más el favorito?", xaxis=dict(title="% de aciertos del favorito", range=[40, g["acierto"].max() + 3]))
            ui.grafico(fig, 520, key="v_fav")
        with c2, st.container(key="tarjeta_v10"):
            st.markdown("### 🤯 Las mayores sorpresas")
            d2 = d.assign(cuota_ganador=np.select([d["resultado"] == "Local", d["resultado"] == "Empate"],
                                                  [d["cuota_local"], d["cuota_empate"]], d["cuota_visitante"]))
            top = d2.nlargest(15, "cuota_ganador")[["fecha", "liga", "local", "goles_local", "goles_visitante", "visitante", "resultado", "cuota_ganador"]]
            st.dataframe(top, hide_index=True, height=470, column_config={
                "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                "goles_local": st.column_config.NumberColumn("GL", width="small"), "goles_visitante": st.column_config.NumberColumn("GV", width="small"),
                "cuota_ganador": st.column_config.NumberColumn("Cuota pagada", format="%.2f")})

    # ================================================================== equipos
    with t4:
        estado = D.estado_equipos()
        equipos_validos = sorted(set(m["local"].astype(str)) | set(m["visitante"].astype(str)))
        defecto = [e for e in ["Barcelona", "Real Madrid", "Liverpool", "Man City"] if e in equipos_validos][:3]
        elegidos = st.multiselect("Compare hasta 4 equipos", equipos_validos, default=defecto or equipos_validos[:2], max_selections=4)
        if elegidos:
            ep = D.equipos_partido()
            ep = ep[ep["equipo"].isin(elegidos) & ep["fecha"].dt.year.between(st.session_state["f_temporadas"][0], st.session_state["f_temporadas"][1] + 1)]
            c1, c2 = st.columns([1.5, 1], gap="medium")
            with c1, st.container(key="tarjeta_v11"):
                fig = go.Figure()
                for i, eq in enumerate(elegidos):
                    s = ep[ep["equipo"] == eq].set_index("fecha")["elo_propio"].resample("MS").last().dropna()
                    fig.add_scatter(x=s.index, y=s.values, name=eq, mode="lines", line=dict(width=2.5),
                                    hovertemplate=f"{eq} · %{{x|%b %Y}}: %{{y:.0f}}<extra></extra>")
                elo = D.elo_clubelo()
                ce = elo[elo["equipo"].isin(elegidos)]
                for eq in elegidos:
                    s = ce[ce["equipo"] == eq]
                    if len(s):
                        fig.add_scatter(x=s["fecha"], y=s["elo"], name=f"{eq} (ClubElo)", mode="lines", line=dict(width=1, dash="dot"),
                                        opacity=0.55, hovertemplate=f"{eq} ClubElo · %{{x|%b %Y}}: %{{y:.0f}}<extra></extra>")
                fig.update_layout(title="Evolución del Elo (línea: Elo propio · punteado: ClubElo)", yaxis_title="Elo", hovermode="x unified")
                ui.grafico(fig, 430, key="v_elo")
            with c2, st.container(key="tarjeta_v12"):
                e = estado.set_index("equipo").loc[[x for x in elegidos if x in estado["equipo"].values]]
                radar = pd.DataFrame({
                    "Elo": (e["elo_propio"] - 1200) / 900, "Forma (pts/5)": e["puntos_prom5"] / 3, "Goles a favor": e["gf_prom5"] / 3,
                    "Defensa": 1 - (e["gc_prom5"] / 3).clip(0, 1), "Pts/partido temp.": e["ppg_temporada"] / 3,
                    "Valor plantilla": np.log1p(e["valor_plantilla"].fillna(0)) / np.log1p(1500)}).clip(0, 1)
                fig = go.Figure()
                for eq, fila in radar.iterrows():
                    fig.add_scatterpolar(r=list(fila.values) + [fila.values[0]], theta=list(radar.columns) + [radar.columns[0]],
                                         fill="toself", name=eq, opacity=0.75)
                fig.update_layout(title="Estado actual (normalizado 0–1)", polar=dict(bgcolor="rgba(0,0,0,0)",
                                  radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,.08)", tickfont=dict(color=ui.MUTED)),
                                  angularaxis=dict(gridcolor="rgba(255,255,255,.08)")))
                ui.grafico(fig, 430, key="v_radar")
            cubo = D.cubos()["equipo_temporada"]
            tabla = cubo[cubo["equipo"].isin(elegidos)].sort_values(["equipo", "temporada_inicio"], ascending=[True, False])
            st.dataframe(tabla[["equipo", "liga", "temporada", "posicion", "partidos", "victorias", "empates", "derrotas", "gf", "gc",
                                "dif_goles", "puntos", "ppg", "elo_final", "valor_plantilla_meur"]], hide_index=True, height=330,
                         column_config={"ppg": st.column_config.NumberColumn("Pts/PJ", format="%.2f"),
                                        "elo_final": st.column_config.NumberColumn("Elo final", format="%.0f"),
                                        "valor_plantilla_meur": st.column_config.NumberColumn("Valor (M€)", format="%.0f"),
                                        "posicion": st.column_config.NumberColumn("Pos.", format="%d")})

    # ================================================================== animadas
    with t5:
        st.caption("Presione ▶ para reproducir la animación por temporada; arrastre el control deslizante para ir a un año.")
        with st.container(key="tarjeta_v13"):
            g = m[m["nivel"] == 1].groupby(["temporada_inicio", "liga"], observed=True)["total_goles"].mean().reset_index()
            ligas_top = m[m["nivel"] == 1]["liga"].value_counts().head(12).index
            g = g[g["liga"].isin(ligas_top)]
            if len(g):
                g = g.sort_values(["temporada_inicio", "total_goles"])
                fig = px.bar(g, x="total_goles", y="liga", animation_frame="temporada_inicio", orientation="h",
                             color="total_goles", color_continuous_scale=ui.SECUENCIAL, range_x=[1.8, 3.6], range_color=[2.0, 3.3],
                             labels={"total_goles": "Goles por partido", "liga": "", "temporada_inicio": "Temporada"})
                fig.update_layout(title="Carrera de ligas goleadoras por temporada", coloraxis_showscale=False)
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600
                ui.grafico(fig, 520, key="v_race")
        with st.container(key="tarjeta_v14"):
            cubo = D.cubos()["equipo_temporada"]
            ligas = sorted(cubo.loc[cubo["valor_plantilla_meur"].notna(), "liga"].unique())
            liga = st.selectbox("Liga para la animación de equipos", ligas, index=ligas.index("Premier League") if "Premier League" in ligas else 0)
            c = cubo[(cubo["liga"] == liga) & cubo["valor_plantilla_meur"].notna()].copy()
            c = c[c["temporada_inicio"].between(*st.session_state["f_temporadas"])].sort_values("temporada_inicio")
            if len(c):
                c["gf"] = c["gf"].clip(lower=1)
                fig = px.scatter(c, x="valor_plantilla_meur", y="ppg", animation_frame="temporada", animation_group="equipo",
                                 size="gf", color="dif_goles", hover_name="equipo", text="equipo", log_x=True, size_max=38,
                                 color_continuous_scale=[[0, "#3987e5"], [0.5, "#383835"], [1, "#22e08a"]], range_y=[0, 3],
                                 range_x=[max(1, c["valor_plantilla_meur"].min() * 0.7), c["valor_plantilla_meur"].max() * 1.5],
                                 labels={"valor_plantilla_meur": "Valor de plantilla (M€, escala log)", "ppg": "Puntos por partido",
                                         "dif_goles": "Dif. goles", "temporada": "Temporada"})
                fig.update_traces(textposition="top center", textfont=dict(size=10, color=ui.MUTED))
                fig.update_layout(title=f"{liga}: ¿la plata compra puntos? (tamaño = goles a favor)")
                ui.grafico(fig, 560, key="v_bubble")
            else:
                st.info("Sin datos de valor de mercado para esa liga en el rango de temporadas.")
