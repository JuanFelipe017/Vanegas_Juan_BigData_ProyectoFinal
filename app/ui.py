"""Componentes visuales y sistema de color del dashboard (HTML/CSS propio + plantilla de Plotly).

Sistema de color: cada color tiene UN significado en toda la app.
- Resultados (solo resultados de partido): Local / Empate / Visitante.
- VERDE = modelo / positivo · ORO = goles · NEUTROS = casa de apuestas y líneas base · ROJO = pérdida / eliminar.
- SECUNDARIOS = entidades sin orden (continentes, equipos, técnicas ETL); nunca se reutilizan los de resultado.
"""
from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
COLORES = {"Local": "#199e70", "Empate": "#d95926", "Visitante": "#3987e5"}
VERDE, ORO, MUTED, TEXTO, ROJO = "#22e08a", "#f4c542", "#8fa89b", "#e7f2ec", "#e66767"
NEUTRO, NEUTRO_OSCURO, TEXTO_OSCURO = "#a3b8ad", "#4d6458", "#04110a"
SECUNDARIOS = ["#9085e9", "#d55181", "#c98500", "#a3b8ad"]
CONTINENTES = {"Europa": "#9085e9", "América": "#d55181", "Asia": "#c98500"}
COLOR_FATIGA = SECUNDARIOS[0]          # variables de fatiga y parón en los gráficos de importancia
# rampas secuenciales: el primer tono sigue siendo visible sobre la tarjeta (≥2:1)
RAMPA_VERDE = ["#245c46", "#2a7a58", "#199e70", "#3cc28c", "#7ddbb0", "#d8f7e7"]
RAMPA_ORO = ["#6b5a1e", "#8a7322", "#b08f27", "#d4a72c", "#f4c542", "#fbe7a6"]
# divergentes: punto medio gris oscuro = "nada"
DIV_LOCAL_VISITANTE = [[0, "#8fbaf3"], [0.25, "#3987e5"], [0.5, "#26302b"], [0.75, "#199e70"], [1, "#7ddbb0"]]
DIV_NEUTRA = [[0, "#d55181"], [0.5, "#3a4a42"], [1, "#22e08a"]]
FONDO_SIN_DATOS = "#07110c"


def texto_sobre(hex_color: str) -> str:
    """Color de texto legible sobre un relleno: oscuro sobre tonos claros, claro sobre tonos oscuros."""
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = sum(c * w for c, w in zip(((x / 12.92) if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in (r, g, b)), (0.2126, 0.7152, 0.0722)))
    return TEXTO_OSCURO if lum > 0.18 else TEXTO


def color_en_rampa(rampa: list[str], t: float) -> str:
    return rampa[min(len(rampa) - 1, max(0, round(t * (len(rampa) - 1))))]


# ==================================================================================================
# Estilos y plantilla de gráficos
# ==================================================================================================
def inyectar_estilos(animaciones: bool) -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    marcador = '<span class="sb-motion-on" style="display:none"></span>' if animaciones else ""
    st.html(f"<style>{css}</style>{marcador}")


def registrar_plantilla(animaciones: bool = True) -> None:
    plantilla = go.layout.Template()
    plantilla.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        separators=",.",                                   # coma decimal y punto de miles (español)
        font=dict(family="Barlow, Segoe UI, sans-serif", color=TEXTO, size=13),
        title=dict(font=dict(family="Barlow Condensed, sans-serif", size=19, color=TEXTO), x=0, xanchor="left", y=0.98, yanchor="top"),
        colorway=SECUNDARIOS,
        xaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor="rgba(255,255,255,0.12)", ticks="",
                   title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED)),
        yaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
                   title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, yanchor="top", x=0, font=dict(color=TEXTO), title_text=""),
        hoverlabel=dict(bgcolor="#0b1a12", bordercolor=NEUTRO_OSCURO, font=dict(family="Barlow, sans-serif", color=TEXTO, size=13)),
        margin=dict(l=10, r=10, t=56, b=10),
        transition=dict(duration=500 if animaciones else 0, easing="cubic-in-out"),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12, tickfont=dict(color=MUTED))),
    )
    pio.templates["sportsbook"] = plantilla
    pio.templates.default = "sportsbook"


def grafico(fig: go.Figure, alto: int = 420, key: str | None = None, **kw) -> None:
    fig.update_layout(height=alto)
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False, "displayModeBar": False}, **kw)


def lollipop(valores, etiquetas, color: str, destacar: int | None = None, formato: str = ".2f", hover: str = "%{y}: %{x}") -> go.Figure:
    """Gráfico de puntos con tallo: honesto cuando el eje no parte de 0 (la longitud no codifica magnitud)."""
    fig = go.Figure()
    for x, y in zip(valores, etiquetas):
        fig.add_shape(type="line", x0=min(valores), x1=x, y0=y, y1=y, line=dict(color="rgba(255,255,255,.12)", width=2), layer="below")
    colores = [VERDE if i == destacar else color for i in range(len(valores))]
    fig.add_scatter(x=list(valores), y=list(etiquetas), mode="markers+text", marker=dict(size=11, color=colores),
                    text=[f"{v:{formato}}".replace(".", ",") for v in valores], textposition="middle right",
                    textfont=dict(color=MUTED, size=11), hovertemplate=hover + "<extra></extra>", showlegend=False)
    return fig


# ==================================================================================================
# Bloques HTML
# ==================================================================================================
def esc(texto) -> str:
    return html.escape(str(texto))


def titulo(texto: str, resaltado: str = "", descripcion: str = "", chips: list[tuple[str, str]] | None = None) -> None:
    chips_html = "".join(f'<span class="sb-chip {clase}"><span class="dot"></span>{esc(t)}</span>' for t, clase in (chips or []))
    st.html(f"""
    <div class="sb-titulo">
      <div>
        <h1>{esc(texto)} <span>{esc(resaltado)}</span></h1>
        {f'<p>{descripcion}</p>' if descripcion else ''}
      </div>
      <div class="sb-chips">{chips_html}</div>
    </div>""")


def kpis(items: list[dict]) -> None:
    """items: {lbl, val, unidad?, sub?, acento?}. La rejilla usa tantas columnas como tarjetas (máx. 6)."""
    tarjetas = []
    for it in items:
        acento = f"--acento:{it['acento']};" if it.get("acento") else ""
        unidad = f"<small>{esc(it['unidad'])}</small>" if it.get("unidad") else ""
        tarjetas.append(f"""
        <div class="sb-kpi" style="{acento}">
          <div class="lbl">{esc(it['lbl'])}</div>
          <div class="val">{esc(it['val'])}{unidad}</div>
          <div class="sub">{it.get('sub', '')}</div>
        </div>""")
    st.html(f'<div class="sb-kpis" style="--n:{min(len(items), 6)}">{"".join(tarjetas)}</div>')


def ticker(items: list[tuple[str, str]]) -> None:
    contenido = "".join(f'<span class="item"><b>{esc(a)}</b>{esc(b)}</span>' for a, b in items)
    st.html(f'<div class="sb-ticker" tabindex="0" aria-label="Titulares de insights; se detiene al pasar el cursor">'
            f'<div class="pista">{contenido}</div></div>')


def barras_probabilidad(probs: dict[str, float]) -> str:
    filas = "".join(
        f'<div class="fila"><span class="nom" style="color:{COLORES[k]}">{k}</span>'
        f'<span class="barra"><i style="--p:{v * 100:.1f}%;--c:{COLORES[k]}"></i></span>'
        f'<span class="pct">{num(v * 100, 1)} %</span></div>' for k, v in probs.items())
    return f'<div class="sb-prob">{filas}</div>'


def insight(n: int, tema: str, titulo_: str, texto: str) -> str:
    return (f'<div class="sb-insight"><div class="n">{n}</div><div class="tema">{esc(tema)}</div>'
            f'<div class="t">{esc(titulo_)}</div><p>{esc(texto)}</p></div>')


def num(x: float, dec: int = 1, signo: bool = False) -> str:
    texto = f"{x:{'+' if signo else ''},.{dec}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def hero_marcador(d: dict, animaciones: bool) -> None:
    """Hero = el hallazgo central (Modelo vs. Casa) sobre la cancha animada. Iframe aislado: su JS no toca la app."""
    st.iframe(f"""
<!doctype html><html lang="es"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@500&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
<style>
  html,body{{margin:0;background:transparent;overflow:hidden;font-family:Barlow,'Segoe UI',sans-serif;color:#e7f2ec}}
  .wrap{{position:relative;height:316px;border-radius:20px;overflow:hidden;border:1px solid rgba(64,224,150,.22);background:#071a11}}
  canvas{{position:absolute;inset:0;width:100%;height:100%}}
  .capa{{position:absolute;inset:0;display:grid;place-items:center;align-content:center;gap:10px;padding:0 24px;text-align:center;
    background:radial-gradient(ellipse at center,rgba(5,13,9,.86) 0%,rgba(5,13,9,.62) 55%,rgba(5,13,9,.25) 100%)}}
  h1{{font:800 clamp(26px,3.6vw,44px)/1 'Barlow Condensed';text-transform:uppercase;margin:0;letter-spacing:.01em}}
  h1 span{{color:#22e08a}}
  .tablero{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:clamp(10px,3vw,28px);margin-top:6px}}
  .eq{{font:700 13px/1.25 'Barlow Condensed';letter-spacing:.14em;text-transform:uppercase;color:#a9c0b4}}
  .eq b{{display:block;font-size:clamp(20px,2.4vw,28px);letter-spacing:.04em;color:#e7f2ec}}
  .eq.izq{{text-align:right}} .eq.izq b{{color:#22e08a}} .eq.der{{text-align:left}}
  .score{{font:700 clamp(40px,7vw,76px)/1 'JetBrains Mono',monospace;padding:.12em .35em;border-radius:14px;background:#020604;
    border:1px solid rgba(244,197,66,.28);color:#f4c542;white-space:nowrap;font-variant-numeric:tabular-nums;box-shadow:0 10px 30px rgba(0,0,0,.45)}}
  .nota{{margin:6px 0 0;color:#a9c0b4;font-size:14px;max-width:62ch}} .nota b{{color:#e7f2ec}} .nota .pos{{color:#22e08a}}
  .fuente{{position:absolute;right:16px;top:14px;font:700 11px 'Barlow Condensed';letter-spacing:.18em;color:#8fa89b;
    background:rgba(5,13,9,.72);padding:6px 10px;border-radius:999px;border:1px solid rgba(143,168,155,.3)}}
  .tablero > *{{min-width:0}} .nota{{overflow-wrap:anywhere}}
  @media (max-width:600px){{ .fuente{{display:none}} .capa{{padding:0 12px;gap:8px}} h1{{font-size:24px}}
    .tablero{{gap:8px}} .eq{{font-size:0}} .eq b{{font-size:14px;letter-spacing:.02em}} .score{{font-size:30px;padding:.1em .22em}}
    .nota{{font-size:12px}} }}
</style></head><body>
<div class="wrap">
 <canvas id="c" aria-hidden="true"></canvas>
 <div class="fuente">DATOS PROCESADOS · PARQUET</div>
 <div class="capa">
   <h1>¿Se le gana <span>a la casa?</span></h1>
   <div class="tablero" role="group" aria-label="Accuracy en prueba: modelo {d['modelo_txt']} por ciento, casa {d['casa_txt']} por ciento">
     <div class="eq izq">{esc(d['modelo_sub'])}<b>Modelo</b></div>
     <div class="score"><span class="n" data-n="{d['modelo']}">{d['modelo_txt']}</span> : <span class="n" data-n="{d['casa']}">{d['casa_txt']}</span></div>
     <div class="eq der">{esc(d['casa_sub'])}<b>Casa</b></div>
   </div>
   <p class="nota">{d['nota']}</p>
 </div>
</div>
<script>
const anim = {str(animaciones).lower()} && !matchMedia('(prefers-reduced-motion: reduce)').matches;
const c = document.getElementById('c'), x = c.getContext('2d');
function tam(){{ c.width = c.clientWidth * devicePixelRatio; c.height = c.clientHeight * devicePixelRatio; }}
tam(); addEventListener('resize', tam);
let t = 0; const trail = [];
function cancha(W,H){{
  for(let i=0;i<14;i++){{ x.fillStyle = i%2 ? 'rgba(255,255,255,.025)' : 'rgba(0,0,0,.05)'; x.fillRect(i*W/14,0,W/14,H); }}
  x.strokeStyle='rgba(255,255,255,.18)'; x.lineWidth=2*devicePixelRatio;
  const m=22*devicePixelRatio; x.strokeRect(m,m,W-2*m,H-2*m);
  x.beginPath(); x.moveTo(W/2,m); x.lineTo(W/2,H-m); x.stroke();
  x.beginPath(); x.arc(W/2,H/2,H*.2,0,Math.PI*2); x.stroke();
  x.strokeRect(W-m-W*.12,H*.24,W*.12,H*.52); x.strokeRect(m,H*.24,W*.12,H*.52);
}}
function cuadro(){{
  const W=c.width,H=c.height; x.clearRect(0,0,W,H); cancha(W,H);
  const bx = W*(.5 + .38*Math.sin(t*.7)), by = H*(.5 + .34*Math.sin(t*1.3+1));
  trail.push([bx,by]); if(trail.length>36) trail.shift();
  for(let i=0;i<trail.length;i++){{ const a=i/trail.length; x.fillStyle=`rgba(244,197,66,${{a*.28}})`;
    x.beginPath(); x.arc(trail[i][0],trail[i][1],(2+a*4)*devicePixelRatio,0,Math.PI*2); x.fill(); }}
  x.fillStyle='#fff'; x.beginPath(); x.arc(bx,by,6*devicePixelRatio,0,Math.PI*2); x.fill();
  t += .01; if(anim) requestAnimationFrame(cuadro);
}}
cuadro();
if(anim){{
  document.querySelectorAll('.score .n').forEach(el=>{{
    const fin=+el.dataset.n, t0=performance.now();
    function paso(ts){{ const p=Math.min(1,(ts-t0)/1400), e=1-Math.pow(1-p,4);
      el.textContent=(fin*e).toFixed(1).replace('.',','); if(p<1) requestAnimationFrame(paso); }}
    requestAnimationFrame(paso);
  }});
}}
</script></body></html>""", height=320)


def panel_titulo(texto: str, resaltado: str) -> None:
    st.html(f'<div class="sb-panel-titulo">{esc(texto)} <span>{esc(resaltado)}</span></div>')
