"""Dashboard Streamlit — Fútbol y apuestas (proyecto final de Big Data).

Lee SOLO resultados ya procesados (Parquet, modelos .joblib y el documento de MongoDB): no repite el ETL ni re-entrena.
Ejecutar:  streamlit run app/app.py   (en Docker lo levanta el servicio `app` en el puerto 8501)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Fútbol & Apuestas · Big Data", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

import datos as D  # noqa: E402
import ui  # noqa: E402

D.iniciar_filtros()
ui.inyectar_estilos(st.session_state["f_animaciones"])
ui.registrar_plantilla(st.session_state["f_animaciones"])


# --------------------------------------------------------------------------------------------------
# Pantalla de espera: el servicio de procesamiento todavía no termina el pipeline
# --------------------------------------------------------------------------------------------------
if not D.listo():
    ui.titulo("El pipeline está", "procesando…",
              "El contenedor <b>procesamiento</b> está descargando los datos, ejecutando el ETL y entrenando los modelos. "
              "Esta página se actualiza sola cada 5 segundos.", [("En proceso", "oro")])

    @st.fragment(run_every=5)
    def esperar():
        if D.listo():
            st.rerun()
        st.progress(0.5, text="Esperando resultados en data/processed y models/ …")

    esperar()
    st.stop()


# --------------------------------------------------------------------------------------------------
# Ventana flotante de opciones (filtros globales que leen todas las páginas)
# --------------------------------------------------------------------------------------------------
def restablecer() -> None:
    for k, v in D.FILTROS_DEFECTO.items():
        if k != "f_animaciones":
            st.session_state[k] = v


partidos = D.partidos()
s = st.session_state
activos = (s["f_temporadas"] != D.FILTROS_DEFECTO["f_temporadas"]) + sum(bool(s[k]) for k in ("f_continentes", "f_paises", "f_ligas", "f_niveles"))
with st.container(key="panel_flotante"):
    with st.popover(f"Opciones · {activos}" if activos else "Opciones", icon=":material/tune:"):
        ui.panel_titulo("Filtros", "globales")
        c1, c2 = st.columns([1.3, 1], vertical_alignment="center")
        c1.html(f'<div class="sb-panel-conteo"><b>{ui.num(len(D.filtrar(partidos)), 0)}</b><span>partidos con estos filtros</span></div>')
        c2.button("Restablecer", on_click=restablecer, width="stretch", icon=":material/restart_alt:", disabled=not activos)
        st.caption("Aplican a todas las pestañas: recortan el cubo por temporada, país, liga y nivel.")
        st.slider("Temporadas (año de inicio)", 2000, 2026, key="f_temporadas")
        continentes = sorted(partidos["continente"].unique())
        st.pills("Continente", continentes, selection_mode="multi", key="f_continentes")
        base = partidos[partidos["continente"].isin(st.session_state["f_continentes"])] if st.session_state["f_continentes"] else partidos
        st.multiselect("Países", sorted(base["pais"].unique()), key="f_paises", placeholder="Todos los países")
        if st.session_state["f_paises"]:
            base = base[base["pais"].isin(st.session_state["f_paises"])]
        st.multiselect("Ligas", sorted(base["liga"].unique()), key="f_ligas", placeholder="Todas las ligas")
        st.segmented_control("Nivel de la liga", [1, 2, 3, 4, 5], selection_mode="multi", key="f_niveles",
                             format_func=lambda n: f"{n}ª")
        st.divider()
        st.toggle("Animaciones", key="f_animaciones", help="Marcador animado, ticker de insights y transiciones de los gráficos")

# --------------------------------------------------------------------------------------------------
# Navegación superior
# --------------------------------------------------------------------------------------------------
from paginas import datos_tabla, insights, modelo, resumen, visualizaciones  # noqa: E402

navegacion = st.navigation([
    st.Page(resumen.pagina, title="Resumen", icon=":material/stadium:", default=True),
    st.Page(datos_tabla.pagina, title="Datos", icon=":material/table_chart:", url_path="datos"),
    st.Page(visualizaciones.pagina, title="Visualizaciones", icon=":material/monitoring:", url_path="visualizaciones"),
    st.Page(modelo.pagina, title="Modelo", icon=":material/model_training:", url_path="modelo"),
    st.Page(insights.pagina, title="Insights y reporte", icon=":material/lightbulb:", url_path="insights"),
], position="top")
navegacion.run()
