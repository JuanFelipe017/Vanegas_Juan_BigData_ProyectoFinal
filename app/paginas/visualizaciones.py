"""Pestaña 3 — Visualizaciones interactivas (todas responden a los filtros del panel flotante)."""
from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import datos as D
import ui

ORDEN = ["Local", "Empate", "Visitante"]


def temporadas_completas(m: pd.DataFrame) -> pd.Series:
    """Temporadas con al menos 60 % de los partidos de una temporada típica (excluye la que está en curso)."""
    n = m.groupby("temporada_inicio").size()
    return n[n >= 0.6 * n.median()].index


def pagina() -> None:
    m = D.filtrar(D.partidos())
    ui.titulo("Lo que dicen", "los datos",
              "Cada gráfica responde una pregunta de negocio. Pase el cursor para ver el detalle, arrastre para hacer zoom y use "
              "<b>Filtros</b> (abajo a la izquierda) para filtrar temporadas, países y ligas.", [(D.descripcion_filtros(), "oro")])
    if m.empty:
        st.info("Los filtros no dejan partidos. Abra **Filtros** y amplíe temporadas, países o ligas.")
        return
    t1, t2, t3, t4, t5, t6 = st.tabs([":material/trending_up: Tendencias", ":material/public: Ligas y países",
                                      ":material/casino: Mercado de apuestas", ":material/battery_alert: Fatiga y parones",
                                      ":material/shield: Equipos", ":material/play_circle: Animadas"])
    completas = temporadas_completas(m)

    # ================================================================== tendencias
    with t1:
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="tarjeta_v1"):
            mc = m[m["temporada_inicio"].isin(completas)]
            pct = pd.crosstab(mc["temporada_inicio"], mc["resultado"], normalize="index")[ORDEN] * 100
            fig = go.Figure()
            for r in ORDEN:
                fig.add_scatter(x=pct.index, y=pct[r], name=r, mode="lines+markers", line=dict(color=ui.COLORES[r], width=3),
                                marker=dict(size=7), hovertemplate=f"{r} · %{{x}}: %{{y:.1f}} %<extra></extra>")
            if 2020 in pct.index:
                fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="rgba(255,255,255,.06)", line_width=0,
                              annotation_text="Estadios vacíos", annotation_position="top left", annotation_font_color=ui.MUTED)
            fig.update_layout(title=f"Ventaja local: {ui.num(pct['Local'].iloc[0], 0)} % → {ui.num(pct['Local'].iloc[-1], 0)} %",
                              yaxis_title="% de partidos", xaxis_title="Temporada completa", hovermode="x unified")
            ui.grafico(fig, 420, key="v_tend1")
        with c2, st.container(key="tarjeta_v2"):
            g = m[m["temporada_inicio"].isin(completas)].groupby("temporada_inicio")["total_goles"].mean()
            fig = go.Figure(go.Scatter(x=g.index, y=g.values, mode="lines+markers", line=dict(color=ui.ORO, width=3), marker=dict(size=7),
                                       name="Goles por partido", hovertemplate="%{x}: %{y:.2f} goles<extra></extra>"))
            fig.update_layout(title=f"Goles por partido: máximo {ui.num(g.max(), 2)} en {g.idxmax()}",
                              yaxis_title="Goles promedio (el eje no parte de 0)", xaxis_title="Temporada completa", showlegend=False)
            ui.grafico(fig, 420, key="v_tend2")
        with st.container(key="tarjeta_v3"):
            tot = m["total_goles"].astype(int).clip(upper=9)
            frec = tot.value_counts(normalize=True).sort_index() * 100
            lam = m["total_goles"].mean()
            fig = go.Figure()
            fig.add_bar(x=frec.index, y=frec.values, name="Observado", marker_color=ui.ORO,
                        hovertemplate="%{x} goles: %{y:.1f} %<extra></extra>")
            fig.add_scatter(x=frec.index, y=[exp(-lam) * lam ** k / factorial(k) * 100 for k in frec.index],
                            name=f"Poisson λ = {ui.num(lam, 2)}", mode="lines+markers", line=dict(color=ui.TEXTO, width=2, dash="dot"))
            fig.update_layout(title="Los goles siguen casi una Poisson", xaxis_title="Goles totales (9 = 9 o más)",
                              yaxis_title="% de partidos", bargap=0.25)
            ui.grafico(fig, 380, key="v_tend3")

    # ================================================================== ligas
    with t2:
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1, st.container(key="tarjeta_v4"):
            g = m.groupby(["continente", "pais", "liga"], observed=True).agg(partidos=("total_goles", "size"),
                                                                                goles=("total_goles", "mean")).reset_index()
            fig = px.sunburst(g, path=["continente", "pais", "liga"], values="partidos", color="goles",
                              color_continuous_scale=ui.RAMPA_ORO, hover_data={"goles": ":.2f"})
            fig.update_traces(marker=dict(line=dict(color="#050d09", width=1.5)),
                              insidetextfont=dict(family="Barlow Condensed", size=13, color=ui.TEXTO_OSCURO))
            fig.update_layout(title="Continente → país → liga · color = goles", coloraxis_colorbar=dict(title="Goles"))
            ui.grafico(fig, 520, key="v_sun")
        with c2, st.container(key="tarjeta_v5"):
            g = m.groupby("liga", observed=True).agg(goles=("total_goles", "mean"), n=("total_goles", "size"))
            g = g[g["n"] >= 200].sort_values("goles")
            fig = ui.lollipop(g["goles"].tolist(), g.index.tolist(), ui.ORO, destacar=len(g) - 1,
                              hover="%{y}: %{x:.2f} goles por partido")
            fig.update_layout(title=f"Goles por liga: {g.index[-1]} es la más goleadora", xaxis_title="Goles promedio por partido")
            ui.grafico(fig, 520, key="v_ligas")
        with st.container(key="tarjeta_v6"):
            primera = m[m["nivel"] == 1] if (m["nivel"] == 1).any() else m
            piv = primera.pivot_table(index="pais", columns="temporada_inicio", values="resultado",
                                      aggfunc=lambda s: (s == "Local").mean() * 100, observed=True)
            piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
            media = (primera["resultado"] == "Local").mean() * 100
            fig = px.imshow(piv, color_continuous_scale=ui.DIV_LOCAL_VISITANTE, color_continuous_midpoint=media,
                            zmin=media - 15, zmax=media + 15, aspect="auto", labels=dict(x="Temporada", y="País", color="% local"))
            fig.update_traces(hovertemplate="%{y} · %{x}: %{z:.1f} % de victorias locales<extra></extra>")
            fig.update_layout(title=f"% de victorias locales por país y temporada · gris = la media ({ui.num(media, 1)} %)",
                              plot_bgcolor=ui.FONDO_SIN_DATOS)
            ui.grafico(fig, 200 + 20 * len(piv), key="v_heat")

    # ================================================================== mercado
    with t3:
        d = m.dropna(subset=["prob_casa_local"])
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="tarjeta_v7"):
            clase = st.pills("Resultado", ORDEN, default="Local", key="v_cal_clase") or "Local"
            p = d[f"prob_casa_{clase.lower()}"]
            bins = pd.cut(p, np.linspace(0, 1, 26))
            t = pd.DataFrame({"p": p, "real": d["resultado"] == clase}).groupby(bins, observed=True).agg(p=("p", "mean"), real=("real", "mean"), n=("p", "size"))
            t = t[t["n"] >= 60]
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, dash="dash", width=1), name="Calibración perfecta", hoverinfo="skip")
            fig.add_scatter(x=t["p"], y=t["real"], mode="lines+markers", name=clase, line=dict(color=ui.COLORES[clase], width=3),
                            marker=dict(size=np.clip(np.sqrt(t["n"]) / 3, 6, 18)), customdata=t["n"],
                            hovertemplate=f"{clase}<br>La casa dice %{{x:.0%}} → pasa %{{y:.0%}}<br>%{{customdata:,}} partidos<extra></extra>")
            fig.update_layout(title=f"¿Están calibradas las casas? ({clase.lower()}: casi perfecto)",
                              xaxis=dict(title="Probabilidad implícita", tickformat=".0%"), yaxis=dict(title="Frecuencia real", tickformat=".0%"))
            ui.grafico(fig, 420, key="v_calib")
        with c2, st.container(key="tarjeta_v8"):
            dm = d[d["temporada_inicio"].isin(completas)]
            g = dm.groupby(["temporada_inicio", "continente"], observed=True)["margen_casa"].mean().mul(100).reset_index()
            fig = px.line(g, x="temporada_inicio", y="margen_casa", color="continente", markers=True, color_discrete_map=ui.CONTINENTES)
            fig.update_traces(line_width=3, hovertemplate="%{x}: %{y:.2f} %<extra>%{fullData.name}</extra>")
            fig.update_layout(title="El margen de la casa no para de bajar", xaxis_title="Temporada completa", yaxis_title="Margen promedio (%)")
            ui.grafico(fig, 460, key="v_margen")
        c1, c2 = st.columns([1, 1.2], gap="medium")
        with c1, st.container(key="tarjeta_v9"):
            g = d.groupby("liga", observed=True).agg(acierto=("acierto_favorito", "mean"), n=("liga", "size"))
            g = g[g["n"] >= 300].sort_values("acierto")
            g["acierto"] = g["acierto"].astype(float) * 100
            fig = ui.lollipop(g["acierto"].tolist(), g.index.tolist(), ui.NEUTRO, formato=".1f",
                              hover="%{y}: el favorito gana %{x:.1f} %")
            fig.update_layout(title=f"El favorito gana más en {g.index[-1]}", xaxis_title="% de aciertos del favorito")
            ui.grafico(fig, 540, key="v_fav")
        with c2, st.container(key="tarjeta_v10"):
            st.markdown("## Las mayores sorpresas")
            d2 = d.assign(cuota_ganador=np.select([d["resultado"] == "Local", d["resultado"] == "Empate"],
                                                  [d["cuota_local"], d["cuota_empate"]], d["cuota_visitante"]))
            top = d2.nlargest(15, "cuota_ganador")[["fecha", "liga", "local", "goles_local", "goles_visitante", "visitante", "resultado", "cuota_ganador"]]
            st.dataframe(top, hide_index=True, height=480, column_config={
                "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"), "liga": "Liga", "local": "Local", "visitante": "Visitante",
                "resultado": "Ganó", "goles_local": st.column_config.NumberColumn("GL", width="small", help="Goles del local"),
                "goles_visitante": st.column_config.NumberColumn("GV", width="small", help="Goles del visitante"),
                "cuota_ganador": st.column_config.NumberColumn("Cuota pagada", format="%.2f")})

    # ================================================================== fatiga y parones
    with t4:
        st.markdown("¿Pesan el cansancio y los viajes de selecciones? Se mide **contra lo que esperaba la casa**: si el mercado ya "
                    "descuenta la fatiga, la diferencia real − esperado queda cerca de 0. Datos de Transfermarkt (minutos de cada "
                    "jugador) y de partidos de selecciones; cubre ligas top europeas desde 2012.")
        f = m.dropna(subset=["prob_casa_local", "dif_viajeros"])
        f = f[f["post_paron"] == 1]
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="tarjeta_v15"):
            if len(f) >= 200:
                cortes = pd.cut(f["dif_viajeros"], [-50, -2.5, -0.5, 0.5, 2.5, 50],
                                labels=["Visitante +3", "Visitante +1-2", "Igual", "Local +1-2", "Local +3"])
                r = f.assign(res=(f["resultado"] == "Local").astype(float) - f["prob_casa_local"]).groupby(cortes, observed=True)
                t = pd.DataFrame({"pp": r["res"].mean() * 100, "n": r.size(), "ee": r["res"].std() / np.sqrt(r.size()) * 100})
                fig = go.Figure(go.Scatter(x=t.index.astype(str), y=t["pp"], mode="markers", marker=dict(size=13, color=ui.COLORES["Local"]),
                                           error_y=dict(type="data", array=1.96 * t["ee"], color=ui.MUTED, thickness=1.5),
                                           customdata=t["n"], hovertemplate="%{x}: %{y:+.1f} pp (%{customdata:,} partidos)<extra></extra>"))
                fig.add_hline(y=0, line_color=ui.MUTED, line_dash="dash")
                fig.update_layout(title="Tras el parón: victorias del local, real − esperado por la casa",
                                  xaxis_title="Quién tuvo más convocados con viaje intercontinental",
                                  yaxis_title="Puntos porcentuales (barra = IC 95 %)")
                ui.grafico(fig, 420, key="v_paron")
            else:
                st.info("Con estos filtros hay muy pocos partidos post-parón con datos de selecciones.")
        with c2, st.container(key="tarjeta_v16"):
            c = m.dropna(subset=["dif_descanso_real", "prob_casa_local"])
            if len(c) >= 500:
                cortes = pd.cut(c["dif_descanso_real"], [-40, -3.5, -1.5, 1.5, 3.5, 40],
                                labels=["Visitante +4 días", "Visitante +2-3", "Igual", "Local +2-3", "Local +4 días"])
                r = c.assign(res=(c["resultado"] == "Local").astype(float) - c["prob_casa_local"]).groupby(cortes, observed=True)
                t = pd.DataFrame({"pp": r["res"].mean() * 100, "n": r.size(), "ee": r["res"].std() / np.sqrt(r.size()) * 100})
                fig = go.Figure(go.Scatter(x=t.index.astype(str), y=t["pp"], mode="markers", marker=dict(size=13, color=ui.COLORES["Local"]),
                                           error_y=dict(type="data", array=1.96 * t["ee"], color=ui.MUTED, thickness=1.5),
                                           customdata=t["n"], hovertemplate="%{x}: %{y:+.1f} pp (%{customdata:,} partidos)<extra></extra>"))
                fig.add_hline(y=0, line_color=ui.MUTED, line_dash="dash")
                fig.update_layout(title="Descanso real (con Champions y copa): real − esperado",
                                  xaxis_title="Quién descansó más desde su partido anterior", yaxis_title="Puntos porcentuales (barra = IC 95 %)")
                ui.grafico(fig, 420, key="v_descanso")
            else:
                st.info("Con estos filtros hay muy pocos partidos con datos de descanso real.")
        st.caption("Si las barras de error cruzan la línea punteada, la diferencia no es distinguible del azar: el mercado ya lo tenía en cuenta.")

    # ================================================================== equipos
    with t5:
        estado = D.estado_equipos()
        equipos_validos = sorted(set(m["local"].astype(str)) | set(m["visitante"].astype(str)))
        defecto = [e for e in ["Barcelona", "Real Madrid", "Liverpool", "Man City"] if e in equipos_validos][:3]
        elegidos = st.multiselect("Compare hasta 4 equipos", equipos_validos, default=defecto or equipos_validos[:2], max_selections=4)
        if elegidos:
            color_eq = {eq: ui.SECUNDARIOS[i] for i, eq in enumerate(elegidos)}
            ep = D.equipos_partido()
            ep = ep[ep["equipo"].isin(elegidos) & ep["fecha"].dt.year.between(st.session_state["f_temporadas"][0], st.session_state["f_temporadas"][1] + 1)]
            c1, c2 = st.columns([1.5, 1], gap="medium")
            with c1, st.container(key="tarjeta_v11"):
                fig = go.Figure()
                for eq in elegidos:
                    s = ep[ep["equipo"] == eq].set_index("fecha")["elo_propio"].resample("MS").last().dropna()
                    fig.add_scatter(x=s.index, y=s.values, name=eq, mode="lines", line=dict(width=2.5, color=color_eq[eq]),
                                    legendgroup=eq, hovertemplate=f"{eq} · %{{x|%b %Y}}: %{{y:.0f}}<extra></extra>")
                elo = D.elo_clubelo()
                for eq in elegidos:
                    s = elo[elo["equipo"] == eq]
                    if len(s):
                        fig.add_scatter(x=s["fecha"], y=s["elo"], name=f"{eq} (ClubElo)", mode="lines", legendgroup=eq,
                                        line=dict(width=1.2, dash="dot", color=color_eq[eq]), opacity=0.55,
                                        hovertemplate=f"{eq} ClubElo · %{{x|%b %Y}}: %{{y:.0f}}<extra></extra>")
                fig.update_layout(title="Evolución del Elo · sólido = Elo propio, punteado = ClubElo", yaxis_title="Elo", hovermode="x unified")
                ui.grafico(fig, 450, key="v_elo")
            with c2, st.container(key="tarjeta_v12"):
                e = estado.set_index("equipo").loc[[x for x in elegidos if x in estado["equipo"].values]]
                radar = pd.DataFrame({
                    "Elo": (e["elo_propio"] - 1200) / 900, "Forma (pts/5)": e["puntos_prom5"] / 3, "Goles a favor": e["gf_prom5"] / 3,
                    "Defensa": 1 - (e["gc_prom5"] / 3).clip(0, 1), "Pts/partido temp.": e["ppg_temporada"] / 3,
                    "Valor plantilla": np.log1p(e["valor_plantilla"].fillna(0)) / np.log1p(1500)}).clip(0, 1)
                fig = go.Figure()
                for eq, fila in radar.iterrows():
                    fig.add_scatterpolar(r=list(fila.values) + [fila.values[0]], theta=list(radar.columns) + [radar.columns[0]],
                                         fill="toself", name=eq, opacity=0.7, line=dict(color=color_eq[eq]))
                fig.update_layout(title="Estado actual (normalizado 0–1)", polar=dict(bgcolor="rgba(0,0,0,0)",
                                  radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,.08)", tickfont=dict(color=ui.MUTED)),
                                  angularaxis=dict(gridcolor="rgba(255,255,255,.08)")))
                ui.grafico(fig, 450, key="v_radar")
            cubo = D.cubos()["equipo_temporada"]
            tabla = cubo[cubo["equipo"].isin(elegidos)].sort_values(["equipo", "temporada_inicio"], ascending=[True, False])
            st.dataframe(tabla[["equipo", "liga", "temporada", "posicion", "partidos", "victorias", "empates", "derrotas", "gf", "gc",
                                "dif_goles", "puntos", "ppg", "elo_final", "valor_plantilla_meur"]], hide_index=True, height=330,
                         column_config={"equipo": "Equipo", "liga": "Liga", "temporada": "Temporada", "partidos": "PJ", "victorias": "G",
                                        "empates": "E", "derrotas": "P", "gf": "GF", "gc": "GC", "dif_goles": "DG", "puntos": "Pts",
                                        "ppg": st.column_config.NumberColumn("Pts/PJ", format="%.2f"),
                                        "elo_final": st.column_config.NumberColumn("Elo final", format="%.0f"),
                                        "valor_plantilla_meur": st.column_config.NumberColumn("Valor (M€)", format="%.0f"),
                                        "posicion": st.column_config.NumberColumn("Pos.", format="%d")})

    # ================================================================== animadas
    with t6:
        st.caption("Presione ▶ para reproducir la animación por temporada; arrastre el control deslizante para ir a un año.")
        with st.container(key="tarjeta_v13"):
            g = m[(m["nivel"] == 1) & m["temporada_inicio"].isin(completas)].groupby(["temporada_inicio", "liga"], observed=True)["total_goles"].mean().reset_index()
            ligas_top = m[m["nivel"] == 1]["liga"].value_counts().head(12).index
            g = g[g["liga"].isin(ligas_top)]
            if len(g):
                g = g.sort_values(["temporada_inicio", "total_goles"])
                fig = px.bar(g, x="total_goles", y="liga", animation_frame="temporada_inicio", orientation="h",
                             color="total_goles", color_continuous_scale=ui.RAMPA_ORO, range_x=[0, 3.6], range_color=[2.0, 3.3],
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
                lim = max(abs(c["dif_goles"].min()), abs(c["dif_goles"].max()))
                fig = px.scatter(c, x="valor_plantilla_meur", y="ppg", animation_frame="temporada", animation_group="equipo",
                                 size="gf", color="dif_goles", hover_name="equipo", text="equipo", log_x=True, size_max=38,
                                 color_continuous_scale=ui.DIV_NEUTRA, range_color=[-lim, lim], range_y=[0, 3],
                                 range_x=[max(1, c["valor_plantilla_meur"].min() * 0.7), c["valor_plantilla_meur"].max() * 1.5],
                                 labels={"valor_plantilla_meur": "Valor de plantilla (M€, escala log)", "ppg": "Puntos por partido",
                                         "dif_goles": "Dif. goles", "temporada": "Temporada"})
                fig.update_traces(textposition="top center", textfont=dict(size=10, color=ui.MUTED))
                fig.update_layout(title=f"{liga}: ¿la plata compra puntos? · tamaño = goles a favor")
                ui.grafico(fig, 560, key="v_bubble")
            else:
                st.info("Sin datos de valor de mercado para esa liga en el rango de temporadas.")
