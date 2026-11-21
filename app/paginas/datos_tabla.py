"""Pestaña 2 — Datos: tabla filtrable, cubos OLAP y explorador multidimensional (slice, dice, drill-down, roll-up)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import datos as D
import ui

DIMENSIONES = {"Continente": "continente", "País": "pais", "Liga": "liga", "Nivel": "nivel", "Década": "decada",
               "Temporada": "temporada", "Resultado": "resultado", "Mes": "mes", "Favorito de la casa": "favorito_casa"}
MEDIDAS = {
    "Partidos": ("resultado", "size", 1),
    "Goles por partido": ("total_goles", "mean", 1),
    "% victorias locales": ("_local", "mean", 100),
    "% empates": ("_empate", "mean", 100),
    "% victorias visitantes": ("_visitante", "mean", 100),
    "% más de 2,5 goles": ("mas_de_2_5", "mean", 100),
    "% ambos marcan": ("ambos_marcan", "mean", 100),
    "Margen de la casa (%)": ("margen_casa", "mean", 100),
    "% acierto del favorito": ("acierto_favorito", "mean", 100),
    "% sorpresas": ("sorpresa", "mean", 100),
    "Amarillas por partido (local)": ("amarillas_local", "mean", 1),
    "Diferencia de Elo promedio": ("diferencia_elo", "mean", 1),
}


def preparar(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    for r in ["local", "empate", "visitante"]:
        m[f"_{r}"] = (m["resultado"] == r.capitalize()).astype(float)
    for c in ["acierto_favorito", "sorpresa", "mas_de_2_5", "ambos_marcan"]:
        m[c] = m[c].astype("float")
    return m


def pagina() -> None:
    m = D.filtrar(D.partidos())
    ui.titulo("Datos limpios y cubos OLAP", "Explora los", "datos",
              "Muestra del dataset limpio (Parquet), las tablas de agregación precalculadas y un explorador para cortar el cubo "
              "por cualquier combinación de dimensiones.", [(f"{ui.num(len(m), 0)} partidos filtrados", ""), (D.descripcion_filtros(), "oro")])
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Partidos", "🧊 Explorador OLAP", "📦 Cubos precalculados", "🧹 Calidad de datos"])

    # ------------------------------------------------------------------ partidos
    with tab1:
        c1, c2, c3 = st.columns([2, 2, 1])
        equipo = c1.text_input("Buscar equipo", placeholder="Ej.: Barcelona, Liverpool, Boca Juniors…")
        resultado = c2.pills("Resultado", ["Local", "Empate", "Visitante"], selection_mode="multi", key="pills_resultado")
        filas = c3.select_slider("Filas", [100, 500, 1000, 5000, 20000], value=1000)
        t = m
        if equipo:
            patron = equipo.strip().lower()
            t = t[t["local"].astype(str).str.lower().str.contains(patron) | t["visitante"].astype(str).str.lower().str.contains(patron)]
        if resultado:
            t = t[t["resultado"].isin(resultado)]
        t = t.sort_values("fecha", ascending=False)
        vista = t[["fecha", "liga", "temporada", "local", "goles_local", "goles_visitante", "visitante", "resultado",
                   "cuota_local", "cuota_empate", "cuota_visitante", "prob_casa_local", "prob_casa_empate", "prob_casa_visitante",
                   "elo_propio_local", "elo_propio_visitante", "valor_plantilla_local", "valor_plantilla_visitante", "sorpresa"]].head(filas)
        st.caption(f"{ui.num(len(t), 0)} partidos coinciden · se muestran los {ui.num(len(vista), 0)} más recientes")
        st.dataframe(vista, hide_index=True, height=520, column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "goles_local": st.column_config.NumberColumn("GL", width="small"), "goles_visitante": st.column_config.NumberColumn("GV", width="small"),
            "cuota_local": st.column_config.NumberColumn("1", format="%.2f"), "cuota_empate": st.column_config.NumberColumn("X", format="%.2f"),
            "cuota_visitante": st.column_config.NumberColumn("2", format="%.2f"),
            "prob_casa_local": st.column_config.ProgressColumn("P(local)", min_value=0, max_value=1, format="percent"),
            "prob_casa_empate": st.column_config.ProgressColumn("P(empate)", min_value=0, max_value=1, format="percent"),
            "prob_casa_visitante": st.column_config.ProgressColumn("P(visitante)", min_value=0, max_value=1, format="percent"),
            "elo_propio_local": st.column_config.NumberColumn("Elo L", format="%.0f"),
            "elo_propio_visitante": st.column_config.NumberColumn("Elo V", format="%.0f"),
            "valor_plantilla_local": st.column_config.NumberColumn("Valor L (M€)", format="%.0f"),
            "valor_plantilla_visitante": st.column_config.NumberColumn("Valor V (M€)", format="%.0f"),
            "sorpresa": st.column_config.CheckboxColumn("¿Sorpresa?"),
        })
        st.download_button("Descargar CSV", vista.to_csv(index=False).encode("utf-8-sig"), "partidos_filtrados.csv", "text/csv",
                           icon=":material/download:")

    # ------------------------------------------------------------------ explorador OLAP
    with tab2:
        st.markdown("Arme su propio corte del cubo: **slice/dice** con el panel de *Opciones*, **drill-down** agregando "
                    "dimensiones a las filas y **roll-up** quitándolas.")
        c1, c2, c3 = st.columns([2, 1.2, 1.4])
        filas_dim = c1.multiselect("Filas (jerarquía para drill-down)", list(DIMENSIONES), default=["País", "Liga"], max_selections=3)
        col_dim = c2.selectbox("Columnas", ["(ninguna)"] + [d for d in DIMENSIONES if d not in filas_dim], index=3)
        medida = c3.selectbox("Medida", list(MEDIDAS), index=2)
        if not filas_dim:
            st.info("Elija al menos una dimensión para las filas.")
        else:
            datos = preparar(m)
            col, func, escala = MEDIDAS[medida]
            claves = [DIMENSIONES[d] for d in filas_dim] + ([DIMENSIONES[col_dim]] if col_dim != "(ninguna)" else [])
            g = datos.groupby(claves, observed=True)[col].agg(func).mul(escala).rename(medida).reset_index()
            g["_n"] = datos.groupby(claves, observed=True).size().to_numpy()
            if func != "size":
                g = g[g["_n"] >= 30]
            if col_dim != "(ninguna)":
                pivote = g.pivot_table(index=[DIMENSIONES[d] for d in filas_dim], columns=DIMENSIONES[col_dim], values=medida, observed=True)
                if col_dim == "Temporada":
                    pivote = pivote.iloc[:, -12:]
                from matplotlib.colors import LinearSegmentedColormap
                escala = LinearSegmentedColormap.from_list("sb", ui.SECUENCIAL)
                estilo = (pivote.style.format(lambda v: "—" if pd.isna(v) else f"{v:,.1f}")
                          .background_gradient(cmap=escala, axis=None, text_color_threshold=0.45)
                          .highlight_null(color="#07110c"))
                st.dataframe(estilo, height=460)
                if len(filas_dim) == 1:
                    fig = px.imshow(pivote, color_continuous_scale=ui.SECUENCIAL, aspect="auto", labels=dict(color=medida))
                    fig.update_layout(xaxis_title=col_dim, yaxis_title=filas_dim[0])
                    ui.grafico(fig, 120 + 22 * len(pivote), key="olap_heat")
            else:
                st.dataframe(g.drop(columns="_n").sort_values(medida, ascending=False), hide_index=True, height=380,
                             column_config={medida: st.column_config.NumberColumn(format="%.1f")})
                g2 = g.copy()
                for c in [DIMENSIONES[d] for d in filas_dim]:
                    g2[c] = g2[c].astype(str)
                fig = px.sunburst(g2, path=[DIMENSIONES[d] for d in filas_dim], values="_n", color=medida,
                                  color_continuous_scale=ui.SECUENCIAL, hover_data={medida: ":.1f", "_n": ":,"})
                fig.update_traces(marker=dict(line=dict(color="#050d09", width=1.5)), insidetextfont=dict(family="Barlow Condensed"))
                fig.update_layout(margin=dict(t=10, b=10), coloraxis_colorbar=dict(title=medida))
                st.caption("Haga clic en un sector para hacer drill-down; clic en el centro para volver (roll-up). Tamaño = partidos.")
                ui.grafico(fig, 560, key="olap_sun")

    # ------------------------------------------------------------------ cubos precalculados
    with tab3:
        cubos = D.cubos()
        nombres = {"Liga × temporada": "liga_temporada", "País × década × nivel × resultado": "pais_decada",
                   "Equipo × temporada (tabla de posiciones)": "equipo_temporada"}
        eleccion = st.segmented_control("Cubo", list(nombres), default="Liga × temporada", key="cubo_sel")
        cubo = cubos[nombres[eleccion or "Liga × temporada"]]
        s = st.session_state
        cubo = cubo[cubo["temporada_inicio"].between(*s["f_temporadas"])] if "temporada_inicio" in cubo else cubo
        if s["f_paises"]:
            cubo = cubo[cubo["pais"].isin(s["f_paises"])]
        if s["f_ligas"] and "liga" in cubo:
            cubo = cubo[cubo["liga"].isin(s["f_ligas"])]
        st.caption(f"{ui.num(len(cubo), 0)} filas (respeta los filtros del panel)")
        st.dataframe(cubo, hide_index=True, height=480)
        st.download_button("Descargar cubo (CSV)", cubo.to_csv(index=False).encode("utf-8-sig"), f"cubo_{nombres[eleccion or 'Liga × temporada']}.csv",
                           "text/csv", icon=":material/download:")

    # ------------------------------------------------------------------ calidad de datos
    with tab4:
        etl = D.resumen_etl()
        st.markdown("#### Problemas detectados en los datos crudos")
        diag = pd.DataFrame(etl["diagnostico"] + [{**d, "porcentaje": None} for d in etl["diagnostico_transfermarkt"]])
        st.dataframe(diag, hide_index=True, column_config={
            "registros": st.column_config.NumberColumn("Registros", format="%d"),
            "porcentaje": st.column_config.ProgressColumn("% del total", min_value=0, max_value=60, format="%.2f %%")})
        st.markdown("#### Las 4 técnicas ETL: conteos antes / después")
        pasos = pd.DataFrame(etl["pasos"])
        colores = {"Limpieza": ui.ROJO, "Normalización": ui.COLORES["Visitante"], "Enriquecimiento": ui.VERDE, "Agregación": ui.ORO}
        fig = go.Figure()
        for tecnica, grupo in pasos.groupby("tecnica", sort=False):
            fig.add_bar(y=grupo["paso"].str.slice(0, 70), x=grupo["afectados"], orientation="h", name=tecnica,
                        marker_color=colores.get(tecnica), hovertemplate="%{y}<br>%{x:,} afectados<extra>" + tecnica + "</extra>")
        fig.update_layout(xaxis_type="log", xaxis_title="Registros afectados (escala logarítmica)",
                          yaxis=dict(autorange="reversed"), legend=dict(y=1.06))
        ui.grafico(fig, 640, key="etl_pasos")
        st.dataframe(pasos, hide_index=True)
