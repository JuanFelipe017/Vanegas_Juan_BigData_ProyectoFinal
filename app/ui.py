"""Componentes visuales reutilizables del dashboard (HTML/CSS propio + plantilla de Plotly)."""
from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
COLORES = {"Local": "#199e70", "Empate": "#d95926", "Visitante": "#3987e5"}
VERDE, ORO, MUTED, TEXTO, ROJO = "#22e08a", "#f4c542", "#8fa89b", "#e7f2ec", "#e66767"
SECUENCIAL = ["#0b1f16", "#0f3d29", "#156b43", "#199e70", "#3cc28c", "#7ddbb0", "#d8f7e7"]
DIVERGENTE = [[0, "#3987e5"], [0.5, "#383835"], [1, "#e66767"]]


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
        font=dict(family="Barlow, Segoe UI, sans-serif", color=TEXTO, size=13),
        title=dict(font=dict(family="Barlow Condensed, sans-serif", size=20, color=TEXTO), x=0, xanchor="left", y=0.97),
        colorway=["#199e70", "#d95926", "#3987e5", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"],
        xaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor="rgba(255,255,255,0.12)", ticks="",
                   title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED)),
        yaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
                   title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXTO)),
        hoverlabel=dict(bgcolor="#0b1a12", bordercolor="#22e08a", font=dict(family="Barlow, sans-serif", color=TEXTO, size=13)),
        margin=dict(l=10, r=10, t=60, b=10),
        transition=dict(duration=600 if animaciones else 0, easing="cubic-in-out"),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))),
    )
    pio.templates["sportsbook"] = plantilla
    pio.templates.default = "sportsbook"


def grafico(fig: go.Figure, alto: int = 420, key: str | None = None, **kw) -> None:
    fig.update_layout(height=alto)
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False, "locale": "es"}, **kw)


# ==================================================================================================
# Bloques HTML
# ==================================================================================================
def esc(texto) -> str:
    return html.escape(str(texto))


def titulo(kicker: str, texto: str, resaltado: str = "", descripcion: str = "", chips: list[tuple[str, str]] | None = None) -> None:
    chips_html = "".join(f'<span class="sb-chip {clase}"><span class="dot"></span>{esc(t)}</span>' for t, clase in (chips or []))
    st.html(f"""
    <div class="sb-titulo">
      <div>
        <div class="kicker">{esc(kicker)}</div>
        <h1>{esc(texto)} <span>{esc(resaltado)}</span></h1>
        {f'<p>{descripcion}</p>' if descripcion else ''}
      </div>
      <div style="display:flex;gap:.5rem;flex-wrap:wrap">{chips_html}</div>
    </div>""")


def kpis(items: list[dict]) -> None:
    """items: {lbl, val, unidad?, sub?, icono?, acento?}"""
    tarjetas = []
    for it in items:
        acento = f"--acento:{it['acento']};" if it.get("acento") else ""
        unidad = f"<small>{esc(it['unidad'])}</small>" if it.get("unidad") else ""
        tarjetas.append(f"""
        <div class="sb-kpi" style="{acento}">
          <div class="ico">{it.get('icono', '')}</div>
          <div class="lbl">{esc(it['lbl'])}</div>
          <div class="val">{esc(it['val'])}{unidad}</div>
          <div class="sub">{it.get('sub', '')}</div>
        </div>""")
    st.html(f'<div class="sb-kpis">{"".join(tarjetas)}</div>')


def marcador(izq: str, izq_sub: str, score: str, der: str, der_sub: str, nota: str) -> None:
    st.html(f"""
    <div class="sb-marcador">
      <div class="eq">{esc(izq_sub)}<b>{esc(izq)}</b></div>
      <div class="score">{esc(score)}</div>
      <div class="eq der">{esc(der_sub)}<b>{esc(der)}</b></div>
      <div class="nota">{nota}</div>
    </div>""")


def ticker(items: list[tuple[str, str]]) -> None:
    contenido = "".join(f'<span class="item"><b>{esc(a)}</b>{esc(b)}</span>' for a, b in items)
    st.html(f'<div class="sb-ticker"><div class="pista">{contenido}</div></div>')


def barras_probabilidad(probs: dict[str, float]) -> str:
    filas = "".join(
        f'<div class="fila"><span class="nom" style="color:{COLORES[k]}">{k}</span>'
        f'<span class="barra"><i style="--p:{v * 100:.1f}%;--c:{COLORES[k]}"></i></span>'
        f'<span class="pct">{v * 100:.1f}%</span></div>' for k, v in probs.items())
    return f'<div class="sb-prob">{filas}</div>'


def insight(n: int, tema: str, titulo_: str, texto: str) -> str:
    return (f'<div class="sb-insight"><div class="n">{n}</div><div class="tema">{esc(tema)}</div>'
            f'<div class="t">{esc(titulo_)}</div><p>{esc(texto)}</p></div>')


def num(x: float, dec: int = 1) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def hero(estadisticas: dict, animaciones: bool) -> None:
    """Cancha animada en canvas con contadores (iframe aislado: su JS no toca el resto de la app)."""
    datos = "".join(
        f'<div class="st"><b data-n="{v}" data-d="{d}">{num(v, d) if not animaciones else "0"}</b><span>{esc(t)}</span></div>'
        for v, d, t in estadisticas["contadores"])
    st.iframe(f"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Barlow:wght@500&display=swap" rel="stylesheet">
<style>
  html,body{{margin:0;background:transparent;overflow:hidden;font-family:Barlow,'Segoe UI',sans-serif;color:#e7f2ec}}
  .wrap{{position:relative;height:300px;border-radius:20px;overflow:hidden;border:1px solid rgba(64,224,150,.25);
    background:linear-gradient(180deg,#0b2a1c,#071a11)}}
  canvas{{position:absolute;inset:0;width:100%;height:100%}}
  .capa{{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;padding:0 38px;
    background:linear-gradient(90deg,rgba(5,13,9,.92) 0%,rgba(5,13,9,.65) 45%,rgba(5,13,9,.05) 75%)}}
  .k{{font:600 12px/1 'Barlow Condensed';letter-spacing:.28em;text-transform:uppercase;color:#f4c542}}
  h1{{font:800 clamp(30px,4.4vw,56px)/.95 'Barlow Condensed';text-transform:uppercase;margin:.35em 0 .25em;letter-spacing:.01em}}
  h1 span{{color:#22e08a}}
  p{{margin:0;color:#a9c0b4;max-width:560px;font-size:15px}}
  .stats{{display:flex;gap:30px;margin-top:22px;flex-wrap:wrap}}
  .st b{{display:block;font:800 30px/1 'Barlow Condensed';color:#e7f2ec;font-variant-numeric:tabular-nums}}
  .st span{{font:600 11px/1.4 'Barlow Condensed';letter-spacing:.16em;text-transform:uppercase;color:#8fa89b}}
  .live{{position:absolute;right:20px;top:18px;display:flex;gap:8px;align-items:center;font:700 12px 'Barlow Condensed';
    letter-spacing:.2em;color:#22e08a;background:rgba(5,13,9,.7);padding:7px 12px;border-radius:999px;border:1px solid rgba(34,224,138,.4)}}
  .live i{{width:8px;height:8px;border-radius:50%;background:#22e08a;box-shadow:0 0 10px #22e08a;animation:l 1.4s infinite}}
  @keyframes l{{50%{{opacity:.25}}}}
</style></head><body>
<div class="wrap"><canvas id="c"></canvas>
 <div class="live"><i></i>{esc(estadisticas['etiqueta'])}</div>
 <div class="capa"><div class="k">{esc(estadisticas['kicker'])}</div>
   <h1>{estadisticas['titulo']}</h1><p>{esc(estadisticas['texto'])}</p><div class="stats">{datos}</div></div></div>
<script>
const anim = {str(animaciones).lower()};
const c = document.getElementById('c'), x = c.getContext('2d');
function tam(){{ c.width = c.clientWidth * devicePixelRatio; c.height = c.clientHeight * devicePixelRatio; }}
tam(); addEventListener('resize', tam);
let t = 0; const trail = [];
function cancha(W,H){{
  for(let i=0;i<14;i++){{ x.fillStyle = i%2 ? 'rgba(255,255,255,.025)' : 'rgba(0,0,0,.05)'; x.fillRect(i*W/14,0,W/14,H); }}
  x.strokeStyle='rgba(255,255,255,.22)'; x.lineWidth=2*devicePixelRatio;
  const m=24*devicePixelRatio; x.strokeRect(m,m,W-2*m,H-2*m);
  x.beginPath(); x.moveTo(W/2,m); x.lineTo(W/2,H-m); x.stroke();
  x.beginPath(); x.arc(W/2,H/2,H*.18,0,Math.PI*2); x.stroke();
  x.strokeRect(W-m-W*.13,H*.25,W*.13,H*.5); x.strokeRect(W-m-W*.05,H*.38,W*.05,H*.24);
  x.strokeRect(m,H*.25,W*.13,H*.5);
}}
function cuadro(){{
  const W=c.width,H=c.height; x.clearRect(0,0,W,H); cancha(W,H);
  // trayectoria del balón: curva de Lissajous sobre la mitad derecha
  const bx = W*(.68 + .22*Math.sin(t*.9)), by = H*(.5 + .32*Math.sin(t*1.7+1));
  trail.push([bx,by]); if(trail.length>40) trail.shift();
  for(let i=0;i<trail.length;i++){{ const a=i/trail.length; x.fillStyle=`rgba(244,197,66,${{a*.35}})`;
    x.beginPath(); x.arc(trail[i][0],trail[i][1],(3+a*5)*devicePixelRatio,0,Math.PI*2); x.fill(); }}
  const g=x.createRadialGradient(bx,by,0,bx,by,40*devicePixelRatio); g.addColorStop(0,'rgba(34,224,138,.35)'); g.addColorStop(1,'rgba(34,224,138,0)');
  x.fillStyle=g; x.beginPath(); x.arc(bx,by,40*devicePixelRatio,0,Math.PI*2); x.fill();
  x.fillStyle='#fff'; x.beginPath(); x.arc(bx,by,8*devicePixelRatio,0,Math.PI*2); x.fill();
  t += .012; if(anim) requestAnimationFrame(cuadro);
}}
cuadro();
if(anim){{
  document.querySelectorAll('.st b').forEach(el=>{{
    const fin=+el.dataset.n, d=+el.dataset.d; const t0=performance.now();
    const f=(n)=>n.toLocaleString('es-CO',{{minimumFractionDigits:d,maximumFractionDigits:d}});
    function paso(ts){{ const p=Math.min(1,(ts-t0)/1600), e=1-Math.pow(1-p,4); el.textContent=f(fin*e); if(p<1) requestAnimationFrame(paso); }}
    requestAnimationFrame(paso);
  }});
}}
</script></body></html>""", height=304)


def panel_titulo(texto: str, resaltado: str) -> None:
    st.html(f'<div class="sb-panel-titulo">⚙️ {esc(texto)} <span>{esc(resaltado)}</span></div>')
