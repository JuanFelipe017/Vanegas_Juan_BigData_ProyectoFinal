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
COLOR_TIPO = {"Línea base": ui.NEUTRO_OSCURO, "Con cuotas": ui.VERDE, "Sin cuotas (fútbol puro)": ui.SECUNDARIOS[0]}
METRICAS = {"log_loss": "Log-loss", "accuracy": "Accuracy", "f1_macro": "F1 macro", "brier": "Brier"}
UMBRAL_VALOR = 0.05          # una sola regla de "apuesta con valor" en toda la página


def pagina() -> None:
    met = D.metricas()
    ui.titulo("El modelo contra", "la casa",
              f"Clasificación Local / Empate / Visitante con partición temporal: se entrena hasta junio de 2023 y se evalúa en "
              f"{ui.num(met['particion']['prueba'], 0)} partidos de 2023/24 en adelante que el modelo nunca vio.",
              [("Modelo cargado desde models/ (sin re-entrenar)", "")])
    with st.expander("¿Cómo leer esta página? (log-loss, Brier, F1, IC 95 %, pp, valor esperado)"):
        st.markdown(
            "- **Log-loss** y **Brier**: miden qué tan buenas son las *probabilidades* (menor es mejor). Son las métricas que importan para apostar.\n"
            "- **F1 macro**: promedio del F1 de Local, Empate y Visitante; obliga a acertar también los empates.\n"
            "- **pp** = puntos porcentuales (de 44 % a 46 % son +2 pp). **IC 95 %**: rango donde estaría el valor real 95 de cada 100 veces; "
            "si cruza el 0, la diferencia puede ser azar.\n"
            "- **Valor esperado (VE)** = probabilidad × cuota − 1: lo que ganarías en promedio por cada peso apostado. "
            "Se considera apuesta con valor si supera el 5 %.")
    t1, t2, t3, t4 = st.tabs([":material/leaderboard: Rendimiento", ":material/psychology: Interpretación",
                              ":material/sports_soccer: Simulador de partido", ":material/payments: Simulador de apuestas"])
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
        metrica = st.segmented_control("Métrica", list(METRICAS), default="log_loss", key="m_metrica", format_func=METRICAS.get) or "log_loss"
        menor_mejor = metrica in ("log_loss", "brier")
        dec = 4 if metrica in ("log_loss", "brier") else 3
        casa = met["lineas_base"]["Favorito de la casa"]["prueba"][metrica]
        elegidos = {f"{res['_mejor']['modelo']} · {NOMBRES[conj].lower()}" for conj, res in met["resultados"].items()}
        # diferencia contra la casa (lo que importa); "Siempre local" queda fuera del eje porque aplastaría la comparación
        t = tabla[tabla["Modelo"] != "Siempre local"].assign(dif=lambda d: d[metrica] - casa).sort_values("dif", ascending=menor_mejor)
        fig = go.Figure()
        for tipo, grupo in t.groupby("Tipo", sort=False):
            fig.add_scatter(x=grupo["dif"], y=grupo["Modelo"], mode="markers+text", name=tipo,
                            marker=dict(size=[17 if mo in elegidos else 12 for mo in grupo["Modelo"]], color=COLOR_TIPO[tipo],
                                        symbol=["star" if mo in elegidos else ("diamond" if tipo == "Línea base" else "circle") for mo in grupo["Modelo"]]),
                            text=[ui.num(v, dec, signo=True) for v in grupo["dif"]], textposition="middle right", textfont=dict(color=ui.MUTED, size=11),
                            customdata=grupo[metrica], hovertemplate="%{y}: %{customdata:." + str(dec) + "f} (%{x:+." + str(dec) + "f} vs. casa)<extra>" + tipo + "</extra>")
        fig.add_vline(x=0, line_dash="dash", line_color=ui.MUTED)
        rango = max(abs(t["dif"].min()), abs(t["dif"].max())) * 1.35
        fig.update_layout(title=f"{METRICAS[metrica]}: diferencia contra la casa",
                          yaxis=dict(categoryorder="array", categoryarray=list(t["Modelo"])[::-1]),
                          xaxis=dict(title=f"{'Negativo' if menor_mejor else 'Positivo'} = mejor que la casa ({ui.num(casa, dec)})", range=[-rango, rango]))
        ui.grafico(fig, 430, key="m_comp")
        ingenua = met["lineas_base"]["Siempre local"]["prueba"][metrica]
        st.caption(f"★ = modelo elegido (por log-loss en **validación**, nunca en prueba; por eso puede no ser el mejor de este gráfico). "
                   f"«Siempre local» queda fuera de escala: {ui.num(ingenua, dec)} ({ui.num(ingenua - casa, dec, signo=True)} vs. la casa).")
    with c2, st.container(key="tarjeta_m2"):
        conj = st.segmented_control("Conjunto", list(NOMBRES), default="con_cuotas", key="m_conj_cm",
                                    format_func={"con_cuotas": "Con cuotas", "sin_cuotas": "Sin cuotas"}.get) or "con_cuotas"
        modo = st.toggle("Usar umbral de empate (elegido en validación)", value=True, key="m_umbral")
        mejor = met["resultados"][conj]["_mejor"]
        r = mejor["prueba_con_umbral"] if modo else met["resultados"][conj][mejor["modelo"]]["prueba"]
        mc = np.array(r["matriz_confusion"])
        pct = mc / mc.sum(axis=1, keepdims=True) * 100
        fig = go.Figure(go.Heatmap(z=pct, x=ORDEN, y=ORDEN, colorscale=ui.RAMPA_VERDE, zmin=0, zmax=80, showscale=False,
                                   hovertemplate="Real %{y} → predicho %{x}: %{z:.1f} %<extra></extra>"))
        for i in range(3):
            for j in range(3):
                fondo = ui.color_en_rampa(ui.RAMPA_VERDE, min(pct[i, j], 80) / 80)
                fig.add_annotation(x=ORDEN[j], y=ORDEN[i], showarrow=False, text=f"<b>{ui.num(pct[i, j], 0)} %</b><br>{ui.num(mc[i, j], 0)}",
                                   font=dict(size=14, family="Barlow Condensed", color=ui.texto_sobre(fondo)))
        fig.update_layout(title=f"Matriz de confusión · {mejor['modelo']}", xaxis_title="Predicción", yaxis=dict(title="Real", autorange="reversed"))
        ui.grafico(fig, 380, key="m_cm")
        st.caption(f"Umbral: predecir empate si p(empate) ≥ {ui.num(mejor['umbral_empate'], 3)}. Sube el recall de empates a cambio de un poco de accuracy.")

    st.dataframe(tabla, hide_index=True, column_config={
        "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.4, max_value=0.55, format="%.4f"),
        "f1_macro": st.column_config.NumberColumn("F1 macro", format="%.4f"), "log_loss": st.column_config.NumberColumn("Log-loss", format="%.4f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.4f"), "roc_auc_ovr": st.column_config.NumberColumn("ROC-AUC", format="%.4f")})
    ce = met["criterios_exito"]
    ui.kpis([
        {"lbl": "Criterio 1 · F1 macro ≥ +10 % vs. ingenua", "val": "Cumple" if ce["cumple_f1_10pct"] else "No cumple",
         "sub": f"<b>{ui.num(ce['f1_macro_vs_ingenua_pct'], 1, signo=True)} %</b>", "acento": ui.VERDE if ce["cumple_f1_10pct"] else ui.ROJO},
        {"lbl": "Criterio 2 · log-loss ≤ casa", "val": "Cumple" if ce["cumple_logloss_casa"] else "No cumple",
         "sub": f"diferencia {ui.num(ce['logloss_modelo_vs_casa'], 5, signo=True)}", "acento": ui.VERDE if ce["cumple_logloss_casa"] else ui.ROJO},
        {"lbl": "Extra · ROI positivo", "val": "Sí" if ce["roi_positivo"] else "No", "acento": ui.VERDE if ce["roi_positivo"] else ui.ROJO,
         "sub": "simulación con valor ≥ 5 % y mejores cuotas"},
        {"lbl": "Fútbol puro vs. casa", "val": f"{ui.num(ce['accuracy_sin_cuotas_vs_casa_pp'], 2, signo=True)} pp", "acento": ui.SECUNDARIOS[0],
         "sub": "accuracy sin usar cuotas"},
    ])


# ==================================================================================================
def interpretacion(met: dict) -> None:
    conj = st.segmented_control("Modelo", list(NOMBRES), default="sin_cuotas", key="m_conj_imp",
                                format_func={"con_cuotas": "Con cuotas", "sin_cuotas": "Sin cuotas"}.get) or "sin_cuotas"
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1, st.container(key="tarjeta_m3"):
        imp = pd.DataFrame(met["importancias"][conj]).head(12).iloc[::-1]
        colores = [ui.COLOR_FATIGA if v.startswith(("descanso_real", "carga_min", "post_paron", "convocados", "viajeros", "dif_carga", "dif_descanso", "dif_viajeros"))
                   else ui.NEUTRO for v in imp["variable"]]
        fig = go.Figure(go.Bar(x=imp["importancia"], y=imp["etiqueta"], orientation="h", marker_color=colores,
                               error_x=dict(type="data", array=imp["desviacion"], color=ui.MUTED, thickness=1),
                               hovertemplate="%{y}: %{x:.5f}<extra></extra>"))
        fig.update_layout(title="Importancia por permutación · violeta = variables de fatiga y parón", xaxis_title="Aumento del log-loss al desordenar la variable")
        ui.grafico(fig, 480, key="m_imp")
    with c2, st.container(key="tarjeta_m4"):
        cal = met["calibracion"]
        clase = st.pills("Clase", ORDEN, default="Local", key="m_cal_clase") or "Local"
        fig = go.Figure()
        fig.add_scatter(x=[0, 0.85], y=[0, 0.85], mode="lines", line=dict(color=ui.NEUTRO_OSCURO, dash="dot", width=1), name="Perfecta", hoverinfo="skip")
        for fuente, nombre, color, guion in [(conj, "Modelo", ui.COLORES[clase], None), ("casa", "Casa (Bet365)", ui.MUTED, "dash")]:
            d = cal[fuente][clase]
            fig.add_scatter(x=d["prob_media"], y=d["frecuencia_real"], mode="lines+markers", name=nombre, line=dict(color=color, width=3, dash=guion),
                            hovertemplate=f"{nombre}: predice %{{x:.0%}} → ocurre %{{y:.0%}}<extra></extra>")
        fig.update_layout(title=f"Calibración en prueba · {clase}", xaxis=dict(title="Probabilidad predicha", tickformat=".0%"),
                          yaxis=dict(title="Frecuencia real", tickformat=".0%"))
        ui.grafico(fig, 480, key="m_cal")
    abl = met.get("ablacion_fatiga")
    if abl:
        st.markdown("#### ¿Aportan la fatiga y los parones? (ablación)")
        st.caption("Mismo LightGBM entrenado con y sin las 12 variables de fatiga y parón. Mejora = log-loss sin fatiga − con fatiga "
                   f"(positivo = la fatiga ayuda). «Tras parón» = los {ui.num(abl['partidos_post_paron_prueba'], 0)} partidos de prueba "
                   "que fueron el primero después de una fecha FIFA.")
        ui.kpis([
            {"lbl": f"{'Con' if c == 'con_cuotas' else 'Sin'} cuotas · {'todos' if k == 'mejora_log_loss' else 'tras parón'}",
             "val": ui.num(abl[c][k], 4, signo=True), "acento": ui.VERDE if abl[c][k] > 0 else ui.ROJO,
             "sub": f"log-loss {ui.num(abl[c]['sin_fatiga'][k.replace('mejora_', '')], 4)} → {ui.num(abl[c]['con_fatiga'][k.replace('mejora_', '')], 4)}"}
            for c in ("sin_cuotas", "con_cuotas") for k in ("mejora_log_loss", "mejora_log_loss_post_paron")])
    top = pd.DataFrame(met["importancias"][conj])
    fatiga = top[top["variable"].str.contains("descanso_real|carga_min|post_paron|convocados|viajeros")]
    st.markdown("#### Lectura en lenguaje de negocio")
    if conj == "sin_cuotas":
        st.markdown(
            f"- **{top.iloc[0]['etiqueta']}** domina: saber cuánto mejor es un equipo que otro (todo su historial resumido en un número) "
            "explica más que cualquier racha reciente.\n"
            f"- Después vienen **{top.iloc[1]['etiqueta']}** y **{top.iloc[2]['etiqueta']}**.\n"
            f"- **Fatiga y parones**: la mejor variable de ese grupo es *{fatiga.iloc[0]['etiqueta'].lower()}* (puesto "
            f"{int(fatiga.index[0]) + 1} de {len(top)}). Aporta, pero mucho menos que el Elo o el valor de plantilla.\n"
            "- La **forma de los últimos 5 partidos** pesa muy poco una vez que se conoce el Elo: las rachas se sobrevaloran.")
    else:
        st.markdown(
            f"- **{top.iloc[0]['etiqueta']}** concentra casi toda la información: la cuota ya resume Elo, forma, lesiones, cansancio y noticias.\n"
            "- Las variables de fútbol y de fatiga añaden poco encima del mercado, pero lo suficiente para encontrar algunas cuotas con "
            "valor (ver *Simulador de apuestas*).\n"
            "- El modelo está tan bien calibrado como la casa: cuando dice 50 %, ocurre ~50 % de las veces.")


# ==================================================================================================
def simulador_partido(met: dict) -> None:
    estado = D.estado_equipos()
    modelos = D.modelos()
    recientes = estado[estado["ultima_fecha"] >= estado["ultima_fecha"].max() - pd.Timedelta(days=420)]
    ligas = recientes.groupby("liga")["equipo"].count()
    ligas = sorted(ligas[ligas >= 8].index.tolist())
    st.markdown("Elija dos equipos de la misma liga. El modelo usa su **último estado conocido** (Elo propio, forma, puntos por "
                "partido, récord como local/visitante y valor de plantilla) y predice sin re-entrenar.")
    c0, c1, c2 = st.columns([1.2, 1, 1])
    liga = c0.selectbox("Liga", ligas, index=ligas.index("LaLiga") if "LaLiga" in ligas else 0)
    equipos = recientes[recientes["liga"] == liga].sort_values("elo_propio", ascending=False)
    nombres = equipos["equipo"].tolist()
    local = c1.selectbox("Local", nombres, index=nombres.index("Real Madrid") if "Real Madrid" in nombres else 0)
    visitante = c2.selectbox("Visitante", [n for n in nombres if n != local], index=0)
    c3, c4, c5 = st.columns(3)
    descanso_l = c3.slider("Días de descanso del local", 2, 14, 7, help="Contando todas las competiciones (liga, Champions, copa)")
    descanso_v = c4.slider("Días de descanso del visitante", 2, 14, 7)
    mes = c5.select_slider("Mes del partido", list(range(1, 13)), value=date.today().month,
                           format_func=lambda x: ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"][x - 1])
    paron = None
    if st.toggle("Es el primer partido después de un parón de selecciones", value=False, key="m_paron"):
        p1, p2, p3, p4 = st.columns(4)
        paron = {"convocados_local": p1.number_input("Convocados del local", 0, 25, 12),
                 "viajeros_local": p2.number_input("…con viaje intercontinental", 0, 25, 4, key="vl"),
                 "convocados_visitante": p3.number_input("Convocados del visitante", 0, 25, 3),
                 "viajeros_visitante": p4.number_input("…con viaje intercontinental", 0, 25, 0, key="vv")}
    usar_cuotas = st.toggle("Tengo las cuotas de una casa (usa el modelo con cuotas)", value=False)
    cuotas = None
    if usar_cuotas:
        q1, q2, q3 = st.columns(3)
        cuotas = (q1.number_input("Cuota local (1)", 1.01, 50.0, 2.10, 0.05), q2.number_input("Cuota empate (X)", 1.01, 50.0, 3.40, 0.05),
                  q3.number_input("Cuota visitante (2)", 1.01, 50.0, 3.60, 0.05))

    fila_l, fila_v = equipos.set_index("equipo").loc[local], equipos.set_index("equipo").loc[visitante]
    paquete = modelos["con_cuotas" if usar_cuotas else "sin_cuotas"]
    base = D.fila_prediccion(fila_l, fila_v, str(fila_l["liga_codigo"]), int(fila_l["nivel"]), mes, descanso_l, descanso_v, cuotas)
    probs = D.predecir(paquete, base)
    probs_paron = D.predecir(paquete, D.fila_prediccion(fila_l, fila_v, str(fila_l["liga_codigo"]), int(fila_l["nivel"]), mes,
                                                        descanso_l, descanso_v, cuotas, paron)) if paron else None
    final = probs_paron or probs

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, st.container(key="tarjeta_m5"):
        st.markdown(f"## {local} <span style='color:var(--muted)'>vs.</span> {visitante}", unsafe_allow_html=True)
        st.caption(f"Modelo: {paquete['nombre']} · {NOMBRES[paquete['conjunto']]} · entrenado hasta {paquete['entrenado_hasta']}")
        st.html(ui.barras_probabilidad(final))
        if probs_paron:
            dl = (probs_paron["Local"] - probs["Local"]) * 100
            st.caption(f"Efecto del parón en la victoria local: **{ui.num(dl, 1, signo=True)} pp** "
                       f"({ui.num(probs['Local'] * 100, 1)} % sin parón → {ui.num(probs_paron['Local'] * 100, 1)} % con estos convocados).")
        justas = {k: 1 / v for k, v in final.items()}
        celdas = []
        for i, k in enumerate(ORDEN):
            ve = final[k] * cuotas[i] - 1 if cuotas else 0.0
            estado = "valor" if ve > UMBRAL_VALOR else ("marginal" if ve > 0 else "")
            etiqueta = {"valor": " · con valor", "marginal": " · marginal", "": ""}[estado]
            extra = (f"<div class='k' style='margin-top:.3rem'>casa {ui.num(cuotas[i], 2)} · VE {ui.num(ve * 100, 1, signo=True)} %{etiqueta}</div>"
                     if cuotas else "")
            celdas.append(f"<div class='sb-cuota {estado}'><div class='k'>Cuota justa · {k}</div>"
                          f"<div class='v'>{ui.num(justas[k], 2)}</div>{extra}</div>")
        st.html(f"<div class='sb-cuotas'>{''.join(celdas)}</div>")
        if cuotas:
            mejores = [k for i, k in enumerate(ORDEN) if final[k] * cuotas[i] - 1 > UMBRAL_VALOR]
            marginales = [k for i, k in enumerate(ORDEN) if 0 < final[k] * cuotas[i] - 1 <= UMBRAL_VALOR]
            if mejores:
                st.success(f"Apuesta con valor según el modelo: **{', '.join(mejores)}** (valor esperado > 5 %).", icon=":material/check_circle:")
            else:
                st.info("Ninguna cuota tiene valor esperado > 5 %: mejor no apostar."
                        + (f" {', '.join(marginales)} queda en zona marginal (0–5 %), dentro del error del modelo." if marginales else ""),
                        icon=":material/block:")
    with c1, st.container(key="tarjeta_m10"):
        calculadora(final, cuotas)
    with c2, st.container(key="tarjeta_m6"):
        comp = pd.DataFrame({
            "Variable": ["Elo propio", "Elo ClubElo", "Puntos prom. (5)", "Goles a favor (5)", "Goles en contra (5)", "Pts/partido temporada",
                         "Récord local / visitante (5)", "Valor plantilla (M€)"],
            local: [fila_l["elo_propio"], fila_l["elo_clubelo"], fila_l["puntos_prom5"], fila_l["gf_prom5"], fila_l["gc_prom5"],
                    fila_l["ppg_temporada"], fila_l["record_local"], fila_l["valor_plantilla"]],
            visitante: [fila_v["elo_propio"], fila_v["elo_clubelo"], fila_v["puntos_prom5"], fila_v["gf_prom5"], fila_v["gc_prom5"],
                        fila_v["ppg_temporada"], fila_v["record_visitante"], fila_v["valor_plantilla"]],
        })
        st.markdown("## Cara a cara")
        menor_es_mejor = [False, False, False, False, True, False, False, False]
        comp["Ventaja"] = [("—" if pd.isna(a) or pd.isna(b) or a == b else (local if (a < b) == m else visitante))
                           for a, b, m in zip(comp[local], comp[visitante], menor_es_mejor)]
        for eq in (local, visitante):
            comp[eq] = [("—" if pd.isna(v) else ui.num(v, 0 if v >= 100 else 2)) for v in comp[eq]]
        st.dataframe(comp, hide_index=True)
        ep = D.equipos_partido()
        h2h = ep[(ep["equipo"] == local) & (ep["rival"] == visitante)].sort_values("fecha", ascending=False).head(8)
        if len(h2h):
          with st.expander(f"Últimos {len(h2h)} enfrentamientos (marcador desde el punto de vista de {local})"):
            st.dataframe(h2h.assign(marcador=h2h["gf"].astype(int).astype(str) + " – " + h2h["gc"].astype(int).astype(str),
                                    condicion=np.where(h2h["es_local"], "Local", "Visitante"))[["fecha", "condicion", "marcador"]],
                         hide_index=True, column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                                         "condicion": "Jugó como", "marcador": "Marcador"})
        st.caption(f"Último partido registrado: {local} {fila_l['ultima_fecha']:%d/%m/%Y} · {visitante} {fila_v['ultima_fecha']:%d/%m/%Y}")


def calculadora(probs: dict[str, float], cuotas: tuple[float, float, float] | None) -> None:
    """¿Cuánto gano si apuesto X? Retorno, ganancia neta y ganancia esperada según el modelo, en la divisa elegida."""
    st.markdown("## ¿Cuánto gano si apuesto?")
    c1, c2 = st.columns(2)
    divisa = c2.selectbox("Divisa", list(ui.DIVISAS), index=0, key="calc_divisa")
    simbolo, dec, ejemplo = ui.DIVISAS[divisa]
    monto = c1.number_input(f"Monto apostado ({simbolo})", min_value=0.0, value=ejemplo, step=10.0 ** max(0, len(str(int(ejemplo))) - 2),
                            format=f"%.{dec}f", key=f"calc_monto_{divisa}")
    apuesta = st.segmented_control("Apuesto a", ORDEN, default="Local", key="calc_apuesta") or "Local"
    cuota = cuotas[ORDEN.index(apuesta)] if cuotas else 1 / probs[apuesta]
    p = probs[apuesta]
    ganancia = monto * (cuota - 1)
    esperado = p * ganancia - (1 - p) * monto
    ui.kpis([
        {"lbl": f"Si gana {apuesta.lower()} recibes", "val": ui.dinero(monto * cuota, divisa), "acento": ui.COLORES[apuesta],
         "sub": f"cuota {ui.num(cuota, 2)} · {'de tu casa' if cuotas else 'justa del modelo'}"},
        {"lbl": "Ganancia neta si aciertas", "val": ui.dinero(ganancia, divisa, signo=True), "acento": ui.VERDE,
         "sub": f"probabilidad según el modelo: {ui.num(p * 100, 1)} %"},
        {"lbl": "Si falla pierdes", "val": ui.dinero(-monto, divisa), "acento": ui.ROJO, "sub": f"probabilidad: {ui.num((1 - p) * 100, 1)} %"},
        {"lbl": "Ganancia esperada", "val": ui.dinero(0 if abs(esperado) < 1e-6 * max(monto, 1) else esperado, divisa, signo=True),
         "acento": None if abs(esperado) < 1e-6 * max(monto, 1) else (ui.VERDE if esperado > 0 else ui.ROJO),
         "sub": "promedio si repitieras esta apuesta muchas veces" + ("" if cuotas else " (0 con cuotas justas)")},
    ], columnas=2)
    q = [cuotas[i] if cuotas else 1 / probs[k] for i, k in enumerate(ORDEN)]
    tabla = pd.DataFrame({"Resultado": ORDEN, "Cuota": [ui.num(x, 2) for x in q],
                          "Prob. modelo": [f"{ui.num(probs[k] * 100, 1)} %" for k in ORDEN],
                          "Recibes si gana": [ui.dinero(monto * x, divisa) for x in q],
                          "Ganancia esperada": [ui.dinero(round(probs[k] * monto * (x - 1) - (1 - probs[k]) * monto, 6), divisa, signo=True)
                                                for k, x in zip(ORDEN, q)]})
    st.dataframe(tabla, hide_index=True)
    st.caption(("Con las cuotas de tu casa, la ganancia esperada es positiva solo si la probabilidad del modelo supera a la que implica la cuota. "
                if cuotas else "Sin cuotas de una casa se usan las cuotas justas del modelo: la ganancia esperada es 0. Activa «Tengo las cuotas» "
                "para ver si una apuesta real conviene. ") + "Cálculo informativo, no es una recomendación de apuesta.")


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
    c1, c2 = st.columns([1.4, 1])
    estrategia = c1.segmented_control("Probabilidades", ["con_cuotas", "sin_cuotas", "casa"], default="con_cuotas", key="ap_estr",
                                      format_func={"con_cuotas": "Modelo con cuotas", "sin_cuotas": "Modelo sin cuotas", "casa": "Control: casa"}.get) or "con_cuotas"
    maximas = c2.toggle("Usar la mejor cuota del mercado", value=True, help="Si se apaga, se usan solo las cuotas de Bet365")
    c3, c4, c5, c6 = st.columns([1.2, 1.2, 0.8, 0.8])
    umbral = c3.slider("Umbral de valor esperado", 0.0, 0.30, 0.05, 0.01, format="%.2f")
    cuota_max = c4.slider("Cuota máxima", 2.0, 15.0, 10.0, 0.5)
    divisa = c6.selectbox("Divisa", list(ui.DIVISAS), index=0, key="ap_divisa")
    simbolo, dec, ejemplo = ui.DIVISAS[divisa]
    unidad = c5.number_input(f"Monto por apuesta ({simbolo})", min_value=0.0, value=ejemplo, format=f"%.{dec}f", key=f"ap_monto_{divisa}")
    cols = ([f"p_{estrategia}_{c.lower()}" for c in ORDEN] if estrategia != "casa" else ["prob_casa_local", "prob_casa_empate", "prob_casa_visitante"])
    apuestas, res = simular_apuestas(pred, cols, umbral, maximas, cuota_max)
    if res["apuestas"] == 0:
        st.info("Con esos parámetros no se hace ninguna apuesta: baje el umbral o suba la cuota máxima.")
        return
    gana = res["beneficio_unidades"] >= 0
    ui.kpis([
        {"lbl": "Apuestas", "val": ui.num(res["apuestas"], 0), "sub": f"de {ui.num(res['partidos_evaluados'], 0)} partidos"},
        {"lbl": "Aciertos", "val": ui.num(res["tasa_acierto"] * 100, 1), "unidad": "%", "sub": f"cuota media {ui.num(res['cuota_media'], 2)}"},
        {"lbl": "Beneficio", "val": ui.dinero(res["beneficio_unidades"] * unidad, divisa, signo=True), "acento": ui.VERDE if gana else ui.ROJO,
         "sub": f"{ui.num(res['beneficio_unidades'], 1, signo=True)} u · apostando {ui.dinero(unidad, divisa)} cada vez "
                f"(total apostado {ui.dinero(res['apuestas'] * unidad, divisa)})"},
        {"lbl": "ROI", "val": ui.num(res["roi"] * 100, 2, signo=True), "unidad": "%", "acento": ui.VERDE if gana else ui.ROJO,
         "sub": "beneficio / unidades apostadas"},
        {"lbl": "Peor racha", "val": ui.dinero(-res["max_caida"] * unidad, divisa), "acento": ui.ROJO,
         "sub": f"{ui.num(res['max_caida'], 1)} u · mayor caída desde el mejor momento"},
    ])
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, st.container(key="tarjeta_m7"):
        fig = go.Figure(go.Scatter(x=apuestas["fecha"], y=apuestas["acumulado"] * unidad, mode="lines", line=dict(color=ui.VERDE if gana else ui.ROJO, width=2.5),
                                   hovertemplate="%{x|%d %b %Y}: %{y:+,." + str(dec) + "f} " + divisa + "<extra></extra>", name="Banca"))
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="Evolución de la banca", yaxis=dict(title=f"Ganancia acumulada ({divisa})", tickformat=",." + str(dec) + "f"), showlegend=False)
        ui.grafico(fig, 400, key="ap_banca")
    with c2, st.container(key="tarjeta_m8"):
        por = apuestas.groupby("apuesta").agg(n=("ganancia", "size"), roi=("ganancia", "mean")).reindex(ORDEN).dropna()
        fig = go.Figure(go.Bar(x=por.index, y=por["roi"] * 100, marker_color=[ui.VERDE if v > 0 else ui.ROJO for v in por["roi"]],
                               text=[f"{ui.num(v * 100, 1, signo=True)} %<br>{ui.num(n, 0)} ap." for v, n in zip(por["roi"], por["n"])],
                               textposition="outside", textfont=dict(color=ui.MUTED), hovertemplate="%{x}: ROI %{y:.2f} %<extra></extra>"))
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="ROI por tipo de apuesta", yaxis_title="ROI (%)")
        ui.grafico(fig, 400, key="ap_tipo")
    curva = pd.DataFrame([simular_apuestas(pred, cols, u, maximas, cuota_max)[1] for u in np.round(np.arange(0, 0.31, 0.02), 2)])
    curva = curva[curva["apuestas"] >= 30]
    with st.container(key="tarjeta_m9"):
        fig = go.Figure()
        fig.add_bar(x=curva["umbral_valor"] * 100, y=curva["apuestas"], name="Apuestas", marker_color="rgba(163,184,173,.22)", yaxis="y2",
                    hovertemplate="Umbral %{x:.0f} %: %{y:,} apuestas<extra></extra>")
        fig.add_scatter(x=curva["umbral_valor"] * 100, y=curva["roi"] * 100, name="ROI", mode="lines+markers", line=dict(color=ui.VERDE, width=3),
                        hovertemplate="Umbral %{x:.0f} %: ROI %{y:+.2f} %<extra></extra>")
        fig.add_hline(y=0, line_color=ui.MUTED, line_width=1)
        fig.update_layout(title="Más exigencia = menos apuestas (y más varianza)", xaxis_title="Umbral de valor esperado (%)",
                          yaxis=dict(title="ROI (%)"), yaxis2=dict(overlaying="y", side="right", showgrid=False, visible=False))
        ui.grafico(fig, 360, key="ap_curva")
        st.caption("Las barras grises (número de apuestas) dan contexto: con pocas apuestas el ROI es pura varianza.")
    st.markdown("#### Últimas apuestas de la simulación")
    st.dataframe(apuestas.sort_values("fecha", ascending=False).head(60), hide_index=True, column_config={
        "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"), "liga_codigo": "Liga", "local": "Local", "visitante": "Visitante",
        "resultado": "Resultado", "apuesta": "Apuesta", "cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
        "prob_modelo": st.column_config.ProgressColumn("Prob.", min_value=0, max_value=1, format="percent"),
        "valor_esperado": st.column_config.NumberColumn("VE", format="percent", help="Valor esperado = probabilidad × cuota − 1"),
        "acierto": "Acierto", "ganancia": st.column_config.NumberColumn("Ganancia", format="%+.2f"),
        "acumulado": st.column_config.NumberColumn("Acumulado", format="%+.1f")})
    st.warning("Simulación académica con datos históricos: no es una recomendación de apuesta. Las casas limitan a quien gana "
               "seguido y las cuotas máximas no siempre están disponibles al mismo tiempo.", icon=":material/info:")
