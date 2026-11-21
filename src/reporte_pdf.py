"""Reporte PDF generado automáticamente a partir del documento guardado en MongoDB (ReportLab).

Uso: ``python src/reporte_pdf.py``  → reports/reporte_futbol_apuestas.pdf
Nada del contenido está escrito a mano: textos, tablas y cifras salen del documento de MongoDB, y las gráficas
son las listadas en ``visualizaciones_generadas`` de ese mismo documento.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# DejaVu (incluida con matplotlib) tiene ≥, −, ×, Δ, →: las fuentes base del PDF no.
_FUENTES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FUENTES / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FUENTES / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

VERDE = colors.HexColor("#0b3d2a")
VERDE_2 = colors.HexColor("#14593d")
DORADO = colors.HexColor("#d4a72c")
GRIS = colors.HexColor("#52514e")
FONDO = colors.HexColor("#f3f6f4")

E = {
    "titulo": ParagraphStyle("titulo", fontName="DejaVu-Bold", fontSize=26, leading=31, textColor=colors.white),
    "subtitulo": ParagraphStyle("subtitulo", fontName="DejaVu", fontSize=13, leading=18, textColor=colors.HexColor("#d8e8df")),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20, textColor=VERDE, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, leading=15, textColor=VERDE_2, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1d1d1b"), spaceAfter=5),
    "pie": ParagraphStyle("pie", fontName="DejaVu", fontSize=7.5, leading=10, textColor=GRIS),
    "celda": ParagraphStyle("celda", fontName="DejaVu", fontSize=7.8, leading=9.6),
    "celda_b": ParagraphStyle("celda_b", fontName="DejaVu-Bold", fontSize=7.8, leading=9.6, textColor=colors.white),
    "kpi_v": ParagraphStyle("kpi_v", fontName="DejaVu-Bold", fontSize=17, leading=20, textColor=VERDE, alignment=TA_CENTER),
    "kpi_t": ParagraphStyle("kpi_t", fontName="DejaVu", fontSize=7.5, leading=9.5, textColor=GRIS, alignment=TA_CENTER),
}


def num(x, dec: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabla(filas: list[list], anchos: list[float], cabecera: bool = True) -> Table:
    datos = [[Paragraph(str(c), E["celda_b"] if (cabecera and i == 0) else E["celda"]) for c in fila] for i, fila in enumerate(filas)]
    t = Table(datos, colWidths=[a * cm for a in anchos], repeatRows=1 if cabecera else 0)
    estilo = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
              ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d9dcd8"))]
    if cabecera:
        estilo += [("BACKGROUND", (0, 0), (-1, 0), VERDE)]
    estilo += [("BACKGROUND", (0, i), (-1, i), FONDO) for i in range(2, len(filas), 2)]
    t.setStyle(TableStyle(estilo))
    return t


def figura(ruta_relativa: str, ancho_cm: float = 16.5) -> Image | Paragraph:
    ruta = C.RAIZ / ruta_relativa
    if not ruta.exists():
        return Paragraph(f"(falta la figura {ruta_relativa})", E["pie"])
    img = Image(str(ruta))
    escala = ancho_cm * cm / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * escala, img.imageHeight * escala
    return img


def _pie_pagina(lienzo, doc):
    lienzo.saveState()
    if doc.page > 1:
        lienzo.setFillColor(VERDE)
        lienzo.rect(0, A4[1] - 0.9 * cm, A4[0], 0.9 * cm, fill=1, stroke=0)
        lienzo.setFillColor(colors.white)
        lienzo.setFont("DejaVu-Bold", 8.5)
        lienzo.drawString(1.6 * cm, A4[1] - 0.58 * cm, "FÚTBOL Y APUESTAS · Pipeline de Big Data con Machine Learning")
        lienzo.setFillColor(DORADO)
        lienzo.drawRightString(A4[0] - 1.6 * cm, A4[1] - 0.58 * cm, "Proyecto final · Big Data")
        lienzo.setFillColor(GRIS)
        lienzo.setFont("DejaVu", 7.5)
        lienzo.drawString(1.6 * cm, 0.9 * cm, f"{C.AUTOR} · Fundación Universitaria Los Libertadores")
        lienzo.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Página {doc.page}")
    lienzo.restoreState()


def _portada(lienzo, doc):
    lienzo.saveState()
    lienzo.setFillColor(VERDE)
    lienzo.rect(0, A4[1] * 0.42, A4[0], A4[1] * 0.58, fill=1, stroke=0)
    # franjas de césped
    lienzo.setFillColor(colors.HexColor("#0e4732"))
    for i in range(0, 12, 2):
        lienzo.rect(i * A4[0] / 12, A4[1] * 0.42, A4[0] / 12, A4[1] * 0.58, fill=1, stroke=0)
    lienzo.setStrokeColor(colors.HexColor("#2f7a57"))
    lienzo.setLineWidth(1.2)
    lienzo.circle(A4[0] * 0.80, A4[1] * 0.66, 2.6 * cm, fill=0, stroke=1)
    lienzo.line(A4[0] * 0.80, A4[1] * 0.42, A4[0] * 0.80, A4[1])
    lienzo.setFillColor(DORADO)
    lienzo.rect(0, A4[1] * 0.42 - 0.18 * cm, A4[0], 0.18 * cm, fill=1, stroke=0)
    lienzo.restoreState()


def kpis(doc_m: dict) -> Table:
    ds = doc_m["metadata"]["dataset"]
    mod = doc_m["modelo"]
    items = [
        (num(ds["registros"], 0), "registros crudos (6 archivos, 3 fuentes)"),
        (num(ds["partidos_limpios"], 0), f"partidos limpios · {ds['ligas']} ligas"),
        (num(mod["metricas"]["accuracy"] * 100, 1) + " %", f"accuracy en prueba ({mod['algoritmo']})"),
        (num(mod["metricas"]["log_loss"], 4), f"log-loss (casa: {num(mod['linea_base']['Favorito de la casa']['log_loss'], 4)})"),
    ]
    celdas = [[Paragraph(v, E["kpi_v"]) for v, _ in items], [Paragraph(t, E["kpi_t"]) for _, t in items]]
    t = Table(celdas, colWidths=[4.3 * cm] * 4)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cfd8d2")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e1e6e2")),
                           ("BACKGROUND", (0, 0), (-1, -1), FONDO), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def generar(doc_m: dict | None = None, salida: Path = C.REPORTE_PDF) -> Path:
    if doc_m is None:
        from mongo_insights import leer
        doc_m = leer()
    if not doc_m:
        raise RuntimeError("No hay documento de análisis en MongoDB: ejecute antes mongo_insights.py")
    meta, ds = doc_m["metadata"], doc_m["metadata"]["dataset"]
    figs = {v["archivo"].split("/")[-1][:2]: v for v in doc_m.get("visualizaciones_generadas", [])}
    h = []

    # ------------------------------------------------------------ portada
    h += [Spacer(1, 3.2 * cm),
          Paragraph("PROYECTO FINAL · GERENCIA Y ALMACENAMIENTO EN BIG DATA", ParagraphStyle("k", parent=E["subtitulo"], fontSize=9, textColor=DORADO)),
          Spacer(1, 0.3 * cm), Paragraph("Fútbol y apuestas:<br/>¿se le puede ganar a la casa?", E["titulo"]), Spacer(1, 0.4 * cm),
          Paragraph(f"Pipeline integral de Big Data con Machine Learning sobre {num(ds['partidos_limpios'], 0)} partidos de "
                    f"{ds['ligas']} ligas ({ds['periodo']}), cuotas de casas de apuestas, ratings Elo y valores de mercado.", E["subtitulo"]),
          Spacer(1, 6.4 * cm), kpis(doc_m), Spacer(1, 0.8 * cm),
          Paragraph(f"<b>Autor:</b> {', '.join(meta['autores'])}<br/><b>Institución:</b> {meta['curso']}<br/>"
                    f"<b>Fecha del análisis:</b> {meta['fecha_analisis'][:10]} · generado automáticamente desde MongoDB el "
                    f"{datetime.now():%Y-%m-%d %H:%M}<br/><b>Fuentes:</b> {ds['fuente']}", E["p"]),
          PageBreak()]

    # ------------------------------------------------------------ 1. dataset
    h.append(Paragraph("1. Resumen del dataset", E["h1"]))
    h.append(Paragraph(f"Se integraron tres fuentes públicas y verificables: resultados, estadísticas y cuotas de "
                       f"<b>Football-Data.co.uk</b>, ratings de <b>ClubElo</b> y valores de mercado de <b>Transfermarkt</b>. En total "
                       f"<b>{num(ds['registros'], 0)} registros crudos</b>; tras el ETL quedan {num(ds['partidos_limpios'], 0)} partidos con "
                       f"{ds['columnas_limpias']} columnas y {num(ds['equipos'], 0)} equipos.", E["p"]))
    filas = [["Archivo", "Fuente", "Filas", "Columnas", "Disco (MB)", "Memoria (MB)"]]
    for nombre, a in ds["archivos"].items():
        filas.append([nombre, a["fuente"].split(" vía")[0], num(a["filas"], 0), a["columnas"], num(a["peso_disco_mb"], 1), num(a["peso_memoria_mb"], 1)])
    h += [tabla(filas, [3.6, 6.4, 1.9, 1.6, 1.7, 1.9]), Spacer(1, 0.3 * cm)]
    est = doc_m["estadisticas_descriptivas"]
    dist = est["distribucion_resultados_pct"]
    h.append(Paragraph(f"Distribución del resultado: Local {num(dist.get('Local'), 1)} %, Empate {num(dist.get('Empate'), 1)} %, "
                       f"Visitante {num(dist.get('Visitante'), 1)} %.", E["p"]))
    filas = [["Variable", "n", "Media", "Desv.", "Mín.", "Mediana", "Máx."]]
    for v in est["variables"]:
        filas.append([v["variable"], num(v["count"], 0), num(v["mean"], 2), num(v["std"], 2), num(v["min"], 2), num(v["50%"], 2), num(v["max"], 2)])
    h += [Paragraph("Estadísticas descriptivas (datos limpios)", E["h2"]), tabla(filas, [4.6, 2, 2, 2, 2, 2, 2])]

    # ------------------------------------------------------------ 2. calidad
    h += [PageBreak(), Paragraph("2. Problemas de calidad resueltos", E["h1"])]
    cal = doc_m["calidad_datos"]
    h.append(Paragraph(f"Se detectaron {len(cal['problemas_detectados'])} tipos de problema; se eliminaron {cal['registros_eliminados']} "
                       "partidos (sin resultado) y el resto de los errores se corrigió anulando solo el valor inválido, para no perder el partido.", E["p"]))
    filas = [["Problema", "Tipo", "Registros", "Acción"]]
    for p in cal["problemas_detectados"]:
        filas.append([p["problema"], p["tipo"], num(p["registros"], 0), p["accion"]])
    h.append(tabla(filas, [6.9, 2.5, 2.1, 5.5]))
    h.append(Paragraph("Las 4 técnicas ETL con conteos antes / después", E["h2"]))
    filas = [["Técnica", "Paso", "Antes", "Después", "Afectados"]]
    for p in cal["pasos_etl"]:
        filas.append([p["tecnica"], p["paso"], num(p["antes"], 0), num(p["despues"], 0), num(p["afectados"], 0)])
    h.append(tabla(filas, [2.7, 8.4, 1.9, 1.9, 2.1]))

    # ------------------------------------------------------------ 3. agregaciones
    h += [PageBreak(), Paragraph("3. Agregaciones multidimensionales (OLAP)", E["h1"])]
    for ag in doc_m["agregaciones"]:
        h.append(Paragraph(f"{ag['nombre']} — {ag['tipo_olap']} ({num(ag['filas_cubo'], 0)} filas en el cubo)", E["h2"]))
        muestra = ag["muestra"][:14]
        if not muestra:
            continue
        columnas = list(muestra[0].keys())
        filas = [[c.replace("_", " ") for c in columnas]]
        for fila in muestra:
            filas.append([num(fila[c], 1) if isinstance(fila[c], (int, float)) and not isinstance(fila[c], bool) else (fila[c] if fila[c] is not None else "—")
                          for c in columnas])
        ancho = 17.0 / len(columnas)
        h += [tabla(filas, [ancho] * len(columnas)), Spacer(1, 0.2 * cm)]
    if "07" in figs:
        h += [KeepTogether([figura(figs["07"]["archivo"], 11.5)])]

    # ------------------------------------------------------------ 4. EDA
    h += [PageBreak(), Paragraph("4. Análisis exploratorio", E["h1"]),
          Paragraph("Cada gráfica responde una pregunta de negocio. El título enuncia el hallazgo; los tres resultados usan siempre los "
                    "mismos colores (validados para daltonismo) y llevan etiqueta directa.", E["p"])]
    for clave, ancho in [("01", 16.5), ("02", 14.5), ("03", 14), ("04", 11.5), ("05", 15.5), ("06", 16.5)]:
        if clave in figs:
            h += [KeepTogether([figura(figs[clave]["archivo"], ancho), Spacer(1, 0.35 * cm)])]

    # ------------------------------------------------------------ 5. modelo
    mod = doc_m["modelo"]
    h += [PageBreak(), Paragraph("5. Modelo de Machine Learning", E["h1"]),
          Paragraph(f"<b>Tarea:</b> {mod['tarea']}. <b>Partición temporal:</b> entrenamiento hasta {mod['particion']['inicio_validacion']}, "
                    f"validación hasta {mod['particion']['inicio_prueba']} y prueba desde entonces ({num(mod['particion']['prueba'], 0)} partidos). "
                    f"Se comparan dos líneas base contra tres algoritmos en dos conjuntos de variables. <b>Mejor modelo con cuotas:</b> "
                    f"{mod['algoritmo']}; <b>sin cuotas:</b> {mod['algoritmo_sin_cuotas']}.", E["p"])]
    filas = [["Modelo", "Accuracy", "F1 macro", "Log-loss", "Brier", "ROC-AUC"]]
    for nombre, r in mod["linea_base"].items():
        filas.append([f"Línea base: {nombre}", num(r["accuracy"] * 100, 1) + " %", num(r["f1_macro"], 3), num(r["log_loss"], 4), num(r["brier"], 4), num(r["roc_auc_ovr"], 3)])
    for conj, res in mod["comparacion"].items():
        for nombre, r in res.items():
            filas.append([f"{nombre} ({conj.replace('_', ' ')})", num(r["accuracy"] * 100, 1) + " %", num(r["f1_macro"], 3), num(r["log_loss"], 4), num(r["brier"], 4), num(r["roc_auc_ovr"], 3)])
    u = mod["metricas_con_umbral_empate"]
    filas.append([f"{mod['algoritmo']} con umbral de empate {num(u['umbral_empate'], 3)}", num(u["accuracy"] * 100, 1) + " %", num(u["f1_macro"], 3), num(u["log_loss"], 4), num(u["brier"], 4), num(u["roc_auc_ovr"], 3)])
    h += [tabla(filas, [6.6, 2, 2, 2, 2, 2.2]), Spacer(1, 0.3 * cm)]
    ce = mod["criterios_exito"]
    h.append(Paragraph(f"<b>Criterios de éxito:</b> F1 macro {num(ce['f1_macro_vs_ingenua_pct'], 1)} % sobre la línea base ingenua "
                       f"({'cumple' if ce['cumple_f1_10pct'] else 'no cumple'} el mínimo de 10 %); log-loss frente a la casa: "
                       f"{num(ce['logloss_modelo_vs_casa'], 5)} ({'igual o mejor' if ce['cumple_logloss_casa'] else 'peor'}); "
                       f"ROI positivo en la simulación: {'sí' if ce['roi_positivo'] else 'no'}.", E["p"]))
    for clave, ancho in [("08", 15.5), ("09", 10), ("10", 15.5), ("11", 15.5)]:
        if clave in figs:
            h += [KeepTogether([figura(figs[clave]["archivo"], ancho), Spacer(1, 0.3 * cm)])]
    filas = [["Variable (modelo con cuotas)", "Importancia"]] + [[v["etiqueta"], num(v["importancia"], 5)] for v in mod["variables_importantes"][:8]]
    h += [Paragraph("Variables más importantes", E["h2"]), tabla(filas, [11, 3])]
    filas = [["Estrategia (umbral 5 %)", "Apuestas", "Acierto", "Cuota media", "Beneficio (u)", "ROI"]]
    nombres = {"con_cuotas": "Modelo con cuotas", "sin_cuotas": "Modelo sin cuotas", "casa": "Control: probabilidades de la casa"}
    for k, s in mod["simulacion_apuestas"].items():
        filas.append([nombres.get(k, k), num(s["apuestas"], 0), num(s["tasa_acierto"] * 100, 1) + " %", num(s["cuota_media"], 2),
                      num(s["beneficio_unidades"], 1), num(s["roi"] * 100, 2) + " %"])
    h += [Paragraph("Simulación de apuestas de valor (temporadas de prueba)", E["h2"]), tabla(filas, [6, 2, 2, 2.2, 2.4, 2.2])]

    # ------------------------------------------------------------ 6. insights y recomendaciones
    h += [PageBreak(), Paragraph("6. Insights", E["h1"])]
    for ins in doc_m["insights"]:
        h.append(KeepTogether([Paragraph(f"{ins['id']}. {ins['titulo']} <font color='#8a8984' size='8'>· {ins['tema']}</font>", E["h2"]),
                               Paragraph(ins["texto"], E["p"])]))
    h.append(Paragraph("7. Recomendaciones", E["h1"]))
    for r in doc_m["recomendaciones"]:
        h.append(Paragraph(f"<b>{r['id']}. {r['dirigida_a']}:</b> {r['texto']}", E["p"]))
    h += [Spacer(1, 0.5 * cm), Paragraph(f"Video de sustentación: {meta.get('video', '')}", E["pie"])]

    pdf = SimpleDocTemplate(str(salida), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="Fútbol y apuestas — Reporte final", author=C.AUTOR,
                            subject="Proyecto final de Big Data")
    pdf.build(h, onFirstPage=_portada, onLaterPages=_pie_pagina)
    return salida


if __name__ == "__main__":
    print("PDF generado:", generar())
