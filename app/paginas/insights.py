"""Pestaña 5 — Insights y reporte: lee el documento de MongoDB y permite descargar el PDF."""
from __future__ import annotations

import streamlit as st

import datos as D
import ui
from config import REPORTE_PDF


def pagina() -> None:
    doc, fuente = D.documento_mongo()
    ui.titulo("Insights · MongoDB · Reporte PDF", "Lo que", "aprendimos",
              "Insights y recomendaciones leídos en vivo desde el documento de análisis guardado en MongoDB, y el reporte PDF "
              "generado automáticamente a partir de ese mismo documento.",
              [(f"Fuente: {fuente}", "" if fuente == "MongoDB" else "rojo")])
    if not doc:
        st.error("No hay documento de análisis. Revise que el servicio de procesamiento haya terminado.")
        return
    if fuente != "MongoDB":
        st.warning("MongoDB no respondió; se muestra la copia JSON del documento (reports/analisis_mongodb.json).", icon="⚠️")

    t1, t2, t3 = st.tabs(["💡 Insights y recomendaciones", "📄 Reporte PDF", "🍃 Documento MongoDB"])
    with t1:
        temas = sorted({i["tema"] for i in doc["insights"]})
        elegidos = st.pills("Filtrar por tema", temas, selection_mode="multi", key="ins_temas")
        lista = [i for i in doc["insights"] if not elegidos or i["tema"] in elegidos]
        c1, c2 = st.columns(2, gap="medium")
        for k, ins in enumerate(lista):
            (c1 if k % 2 == 0 else c2).html(ui.insight(ins["id"], ins["tema"], ins["titulo"], ins["texto"]))
        st.markdown("### 🧭 Recomendaciones")
        for r in doc["recomendaciones"]:
            st.html(f'<div class="sb-insight"><div class="n">R{r["id"]}</div><div class="tema">{ui.esc(r["dirigida_a"])}</div>'
                    f'<p style="margin-top:.1rem">{ui.esc(r["texto"])}</p></div>')
    with t2:
        if REPORTE_PDF.exists():
            datos_pdf = REPORTE_PDF.read_bytes()
            c1, c2 = st.columns([1, 2])
            c1.download_button("Descargar reporte PDF", datos_pdf, REPORTE_PDF.name, "application/pdf", type="primary",
                               icon=":material/picture_as_pdf:", width="stretch")
            c2.caption(f"{REPORTE_PDF.name} · {len(datos_pdf) / 1e6:.2f} MB · generado desde MongoDB por src/reporte_pdf.py")
            vista_previa(datos_pdf)
        else:
            st.info("El PDF todavía no existe en reports/.")
    with t3:
        meta = doc["metadata"]
        ui.kpis([
            {"lbl": "Base / colección", "val": "futbol_apuestas", "sub": "colección <b>analisis</b>", "icono": "🍃"},
            {"lbl": "Fecha del análisis", "val": meta["fecha_analisis"][:10], "sub": meta["fecha_analisis"][11:19] + " UTC", "icono": "🗓️", "acento": ui.ORO},
            {"lbl": "Insights", "val": str(len(doc["insights"])), "sub": f"{len(doc['recomendaciones'])} recomendaciones", "icono": "💡"},
            {"lbl": "Problemas de calidad", "val": str(len(doc["calidad_datos"]["problemas_detectados"])), "icono": "🧹", "acento": ui.ROJO,
             "sub": f"{doc['calidad_datos']['registros_eliminados']} partidos eliminados"},
        ])
        st.markdown("#### Consultas en vivo")
        if fuente == "MongoDB":
            try:
                import sys
                from pymongo import MongoClient

                import config as C
                from mongo_insights import consultas_ejemplo
                cliente = MongoClient(C.MONGO_URI, serverSelectionTimeoutMS=2500)
                resultados = consultas_ejemplo(cliente[C.MONGO_DB][C.MONGO_COLECCION])
                cliente.close()
                for nombre, valor in resultados.items():
                    with st.expander(nombre.replace("_", " ").capitalize()):
                        st.json(valor, expanded=2)
            except Exception as error:  # noqa: BLE001
                st.warning(f"No se pudieron ejecutar las consultas: {error}")
        else:
            st.caption("Las consultas en vivo necesitan MongoDB.")
        st.markdown("#### Documento completo")
        st.json(doc, expanded=1)


def vista_previa(datos_pdf: bytes) -> None:
    try:
        st.pdf(datos_pdf, height=820)
        return
    except Exception:  # noqa: BLE001 - st.pdf requiere el extra streamlit[pdf]; si falta, se renderizan páginas
        pass
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(datos_pdf)
        pagina = st.slider("Página", 1, len(pdf), 1)
        st.image(pdf[pagina - 1].render(scale=1.6).to_pil(), width="stretch")
    except Exception as error:  # noqa: BLE001
        st.info(f"Vista previa no disponible ({error}); use el botón de descarga.")
