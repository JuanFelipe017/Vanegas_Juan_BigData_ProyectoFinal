# ⚽ Fútbol y apuestas: ¿se le puede ganar a la casa?

**Proyecto final — Gerencia y almacenamiento en Big Data · Fundación Universitaria Los Libertadores**
**Autor:** Juan Felipe Vanegas Silva
**Video de sustentación:** https://youtu.be/0VBYjifZxCE
**Dashboard en línea:** <https://juanfelipe017-vanegas-juan-bigdata-proyectofinal-appapp-jlbtk7.streamlit.app> (Streamlit Community Cloud; sin MongoDB, lee la copia JSON del documento)

Pipeline integral de Big Data con Machine Learning sobre **3.253.548 registros** de cuatro fuentes públicas:
**238.858 partidos de 38 ligas (2000–2026)** con resultados, estadísticas y cuotas de casas de apuestas
(Football-Data.co.uk), **273.972 ratings Elo** (ClubElo), **656.301 valoraciones** y **1.894.350 minutos de jugadores**
(Transfermarkt) y **50.099 partidos de selecciones** (martj42) para medir fatiga y parones. Incluye ETL con CRISP-DM,
cubos OLAP, un modelo que compite contra las casas de apuestas, insights en MongoDB, un reporte PDF automático y un
dashboard Streamlit, todo en Docker.

> **Pregunta de negocio:** ¿se puede predecir el resultado de un partido con datos públicos tan bien como las casas de
> apuestas, y alcanza eso para obtener beneficio apostando?

| Resultado principal (29.190 partidos de prueba, 2023/24 en adelante) | Valor |
|---|---|
| Línea base "siempre gana el local": accuracy / log-loss | 43,4 % / 1,0759 |
| Casa de apuestas (Bet365, favorito): accuracy / log-loss | 50,2 % / 1,0029 |
| Mejor modelo **con cuotas** (Regresión logística): accuracy / log-loss | **50,2 % / 1,0032** (empate técnico: +0,0002 vs. la casa) |
| Mejor modelo **sin cuotas** (LightGBM, solo datos públicos): accuracy | 49,0 % |
| F1 macro con umbral de empate vs. línea base ingenua | **+121 %** |
| ROI apostando con valor esperado ≥ 5 % (mejor cuota del mercado) | **+0,73 %** (control con probabilidades de la casa: -3,46 %) |
| Fatiga y parones (ablación): mejora de log-loss en todos / tras un parón | sin cuotas +0,0001 / **+0,0016** · con cuotas +0,0001 / **+0,0014** |

---

## 1. Arquitectura

```
                 datos abiertos (GitHub raw · Cloudflare R2)
                                │  src/descargar_datos.py
┌───────────────────────────────┼──── docker compose up --build ─────────────────────────────────┐
│                               ▼                                                                 │
│  ┌──────────────────────────────────┐   insights    ┌──────────────┐   lee    ┌──────────────────┐ │
│  │ procesamiento  (Python 3.13)     │──────────────▶│   mongo:8.0  │◀─────────│ app (Streamlit)  │ │
│  │ pipeline.py: descarga → ETL →    │   documento   │  colección   │          │ puerto 8501      │ │
│  │ ML → gráficas → Mongo → PDF      │     JSON      │  analisis    │          │ 5 pestañas +     │ │
│  │ + JupyterLab (puerto 8888)       │               └──────────────┘          │ panel flotante   │ │
│  └───────────────┬──────────────────┘                     ▲ volumen            └────────▲─────────┘ │
│                  │ escribe                                 │ mongo_data                  │ lee (ro)  │
│                  ▼                                                                       │           │
│      volúmenes: data/raw · data/processed · models/ · reports/ ─────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Servicio | Imagen | Qué hace | Puerto |
|---|---|---|---|
| `mongo` | `mongo:8.0` | Guarda el documento de análisis (insights, métricas, metadatos). Volumen `mongo_data` | 27018 (anfitrión) → 27017 |
| `procesamiento` | `procesamiento/Dockerfile` | Ejecuta `src/pipeline.py` (idempotente) y deja **JupyterLab** abierto para el notebook | 8888 |
| `app` | `app/Dockerfile` | Dashboard Streamlit. Solo **lee** Parquet, modelos y MongoDB: no repite el ETL ni re-entrena | 8501 |

**Orden de arranque (automático):** `mongo` (healthcheck `ping`) → `procesamiento` (healthcheck: existe
`reports/.pipeline_ok`) → `app`. La app no abre hasta que el pipeline terminó.

## 2. Requisitos

- **Docker Desktop** (Windows/macOS) o Docker Engine + Compose v2 (Linux). Probado con Docker 29 y Compose v5.
- ~3 GB libres en disco (imágenes + datos) y ≥ 6 GB de RAM asignados a Docker.
- Conexión a internet la primera vez (descarga de imágenes y de ~70 MB de datos).

## 3. Cómo ejecutar (un solo comando)

```bash
cd Vanegas_Juan_BigData_ProyectoFinal
cp .env.example .env        # opcional: cambiar la clave de MongoDB y los puertos
docker compose up --build
```

- **Primera vez:** construye las imágenes (~3–5 min), descarga los datos crudos a `data/raw/` y ejecuta el pipeline. Como
  la entrega ya incluye el Parquet limpio y los modelos entrenados, el pipeline **no** re-entrena: solo carga MongoDB y
  genera el PDF (~1 min). Para recalcular todo desde los datos crudos:
  `docker compose run --rm procesamiento python src/pipeline.py --forzar` (~7 min).
- Sin `.env`, `docker compose` usa los valores por defecto de `docker-compose.yml`.

| Qué | URL |
|---|---|
| **Dashboard** | <http://localhost:8501> |
| JupyterLab (token `bigdata`, configurable en `.env`) | <http://localhost:8888/lab?token=bigdata> |
| MongoDB (desde el anfitrión, p. ej. Compass) | `mongodb://admin:<clave>@localhost:27018/?authSource=admin` |

### Cómo verificar que todo funciona
1. `docker compose ps` → `futbol-mongo` y `futbol-procesamiento` **healthy**, `futbol-app` **healthy**.
2. `docker compose logs procesamiento` termina con `✔ Pipeline completo … Dashboard: http://localhost:8501`.
3. En el dashboard, pestaña **Insights y reporte**: la etiqueta debe decir **Fuente: MongoDB** y el botón descarga
   `reporte_futbol_apuestas.pdf`.
4. Consulta directa a Mongo:
   `docker compose exec mongo mongosh -u admin -p <clave> --quiet --eval "db.getSiblingDB('futbol_apuestas').analisis.countDocuments()"` → `1`.

### Ejecutar el notebook dentro del contenedor (Restart & Run All)
Desde JupyterLab (<http://localhost:8888>) abrir `notebooks/Vanegas_Juan_BigData_ProyectoFinal.ipynb` → *Kernel → Restart
Kernel and Run All Cells*, o por consola:

```bash
docker compose exec procesamiento jupyter nbconvert --to notebook --execute --inplace notebooks/Vanegas_Juan_BigData_ProyectoFinal.ipynb --ExecutePreprocessor.timeout=1800
```

El notebook vuelve a ejecutar el ETL y el entrenamiento (~12 min) y escribe en MongoDB, así que la entrega queda
reproducible de principio a fin.

### Benchmark pandas vs. Spark (opcional, fuera de Docker)
Spark no se incluyó en las imágenes (la justificación está en la sección 2 del notebook). El benchmark se corre en el
equipo anfitrión con Java 17 y `pip install pyspark pandas pyarrow psutil`:

```bash
python src/benchmark_spark.py
```

Escribe `reports/benchmark_pandas_spark.json`, que el notebook muestra en la sección 2.

### Apagar
`docker compose down` (conserva los datos de MongoDB) · `docker compose down -v` (borra también el volumen de Mongo).

## 4. Estructura

```
Vanegas_Juan_BigData_ProyectoFinal/
├── README.md
├── docker-compose.yml          # 3 servicios, healthchecks y volúmenes
├── .env.example                # credenciales y puertos de ejemplo (nunca las reales)
├── data/
│   ├── raw/                    # vacía en la entrega; la llena src/descargar_datos.py
│   └── processed/              # Parquet limpio, formato largo, cubos OLAP, predicciones, resumen del ETL
├── notebooks/
│   └── Vanegas_Juan_BigData_ProyectoFinal.ipynb   # CRISP-DM completo, ejecutado
├── src/
│   ├── config.py               # rutas relativas, fuentes, diccionario de ligas, MongoDB desde variables de entorno
│   ├── descargar_datos.py      # descarga + registro (URL, fecha, filas, columnas, peso)
│   ├── etl.py                  # diagnóstico y las 4 técnicas ETL con conteos antes/después
│   ├── modelo.py               # líneas base, 3 algoritmos × 2 conjuntos de variables, evaluación
│   ├── apuestas.py             # simulación de apuestas de valor (la usa también el dashboard)
│   ├── graficas.py             # 11 figuras estáticas con principios de la Actividad 7
│   ├── mongo_insights.py       # documento de análisis, guardado y consultas
│   ├── reporte_pdf.py          # PDF automático (ReportLab) leído desde MongoDB
│   ├── benchmark_spark.py      # pandas vs. Spark sobre los mismos datos
│   └── pipeline.py             # orquestador idempotente (lo ejecuta Docker)
├── tests/                      # test de la lógica de dinero (python tests/test_apuestas.py)
├── models/                     # modelo_con_cuotas.joblib, modelo_sin_cuotas.joblib, metricas_modelos.json
├── reports/                    # reporte_futbol_apuestas.pdf, figuras/, analisis_mongodb.json
├── procesamiento/              # Dockerfile + requirements del servicio de procesamiento
└── app/
    ├── Dockerfile
    ├── app.py                  # navegación + ventana flotante de opciones (filtros globales)
    ├── ui.py · datos.py        # componentes visuales y carga de datos/modelos
    ├── paginas/                # resumen, datos, visualizaciones, modelo, insights
    ├── assets/style.css        # tema "sportsbook / estadio" con animaciones
    └── .streamlit/config.toml
```

## 5. El dashboard

Tema oscuro tipo *sportsbook/estadio* (césped con franjas, tipografía de marcador). El hero es el hallazgo central: el
marcador **Modelo : Casa** sobre una cancha animada. El botón verde **Filtros** (abajo a la **izquierda**, para no chocar con
los controles de Streamlit Cloud) abre una **ventana flotante** con los filtros globales y muestra cuántos están activos.

**Sistema de color** (revisado con la skill de diseño Impeccable): cada color tiene un único significado. Local `#199e70`,
Empate `#d95926` y Visitante `#3987e5` solo para resultados (seguros para daltonismo); verde de marca = modelo/positivo; oro =
goles; neutros = casa y líneas base; violeta/magenta/ocre = entidades (continentes, equipos). Escalas visibles sobre fondo
oscuro, divergente centrada en la media para % de victorias locales, coma decimal en todos los gráficos.

| Pestaña | Contenido |
|---|---|
| **Resumen** | Cancha animada con contadores, ticker de insights, KPIs del dataset filtrado, marcador Modelo vs. Casa, KPIs del modelo |
| **Datos** | Tabla filtrable (búsqueda de equipo, resultado), **explorador OLAP** (pivote con drill-down/roll-up y *sunburst*), cubos precalculados descargables y evidencia de calidad |
| **Visualizaciones** | Tendencias, ligas y países (sunburst, mapa de calor), mercado de apuestas (calibración, margen, sorpresas), **fatiga y parones** (real − esperado según viajeros y descanso real), comparador de equipos (Elo, radar, tabla), animaciones (carrera de ligas, burbujas valor vs. puntos) |
| **Modelo** | Comparación de modelos, matriz de confusión, importancia, calibración y **ablación de fatiga**; **simulador de partido** (predicción nueva con el modelo guardado, con opción de parón: convocados y viajeros, y **calculadora de ganancias**: monto en COP, USD, EUR u otra divisa → cuánto recibes, ganancia neta y ganancia esperada) y **simulador de apuestas** con umbral ajustable y beneficio en la divisa elegida |
| **Insights y reporte** | Insights y recomendaciones leídos de MongoDB, filtro por tema, consultas en vivo, documento completo y **descarga/vista previa del PDF** |

## 6. Fuentes de datos

| Archivo | Fuente | Filas |
|---|---|---|
| `Matches.csv` | [Football-Data.co.uk](https://www.football-data.co.uk/data.php) vía [xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data) | 238.858 |
| `EloRatings.csv` | [ClubElo](http://clubelo.com/) vía el mismo repositorio | 273.972 |
| `games.csv.gz`, `clubs.csv.gz`, `competitions.csv.gz`, `player_valuations.csv.gz` | Transfermarkt vía [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | 88.958 · 796 · 65 · 656.301 |
| `appearances.csv.gz`, `players.csv.gz` | Transfermarkt (minutos por jugador y partido en todas las competiciones; nacionalidad e internacionalidades) | 1.894.350 · 50.149 |
| `selecciones_results.csv` | [martj42/international_results](https://github.com/martj42/international_results) (partidos de selecciones hasta nov-2026) | 50.099 |

El registro exacto (URL, fecha de descarga, filas, columnas, MB en disco y en memoria) queda en
`data/raw/registro_descarga.json` y se muestra en el notebook.

## 7. Problemas de infraestructura encontrados (diagnóstico y solución)

### 7.1 Docker Desktop no arrancaba: sockets AF_UNIX huérfanos
**Síntoma:** `docker info` → `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` y luego el
diálogo *"An unexpected error occurred"*:
```
starting services: initializing Ingest server: listening on unix://C:/Users/<usuario>/AppData/Local/Docker/run/sailor-ingest.sock:
remove …/sailor-ingest.sock: The file cannot be accessed by the system.
```
**Diagnóstico:** el log `%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log` mostraba `backend crashed`. Docker
Desktop usa sockets de Unix (AF_UNIX) también en Windows; tras un apagado no limpio quedaron archivos de socket
(*reparse points*) que ni Docker ni `Remove-Item -Force` podían borrar. Al renombrar la carpeta `run` apareció el mismo error
en otro socket: `%LOCALAPPDATA%\docker-secrets-engine\engine.sock`. Además, al lanzar Docker Desktop desde una terminal
restringida (sandbox de la herramienta de desarrollo) los sockets nuevos quedaban igual de inaccesibles.
**Solución:** cerrar Docker Desktop (*Quit*, **no** *Reset to factory defaults*, que borra imágenes y volúmenes),
`wsl --shutdown`, borrar desde una PowerShell normal `%LOCALAPPDATA%\Docker\run` y `%LOCALAPPDATA%\docker-secrets-engine`
(reiniciar el PC si siguen bloqueados) y abrir Docker Desktop desde el menú Inicio.

### 7.2 Choque de puertos con contenedores de actividades anteriores
**Síntoma:** al planear el despliegue, `docker ps` mostró contenedores de la Actividad 1 todavía activos:
`mongodb 0.0.0.0:27017->27017`, `mysql 3306` y `postgres 5432`. Publicar el Mongo del proyecto en 27017 haría fallar
`docker compose up` con `Bind for 0.0.0.0:27017 failed: port is already allocated`.
**Diagnóstico:** dentro de la red de Compose los servicios se hablan por nombre (`mongo:27017`), así que el puerto del
anfitrión solo sirve para conectarse desde fuera (Compass, mongosh local). No hace falta que sea 27017.
**Solución:** el puerto del anfitrión es configurable (`MONGO_PUERTO_HOST`, `APP_PUERTO_HOST`, `JUPYTER_PUERTO_HOST` en
`.env`) y por defecto Mongo se publica en **27018**, para no chocar con un Mongo local o de otras actividades.

### 7.3 El CDN de Transfermarkt respondía 403 a Python
**Síntoma:** `curl` descargaba `games.csv.gz`, pero `urllib.request.urlopen` fallaba con `HTTP Error 403: Forbidden`.
**Diagnóstico:** misma URL y misma red, solo cambiaba el cliente: el CDN (Cloudflare R2) bloquea el *User-Agent* por
defecto `Python-urllib/3.x`.
**Solución:** `src/descargar_datos.py` envía un *User-Agent* descriptivo que identifica al proyecto (sin suplantar un
navegador), con reintentos y escritura atómica (`.part` → archivo final).

### 7.4 football-data.co.uk cerraba la conexión
**Síntoma:** `curl: (35) Recv failure: Connection was reset` al seguir la redirección a `football-data.co.uk`.
**Solución:** usar el espejo público en GitHub que reúne los CSV de Football-Data.co.uk (y ClubElo) con columnas
homogéneas; la fuente original queda citada.

### 7.5 Spark no arrancaba en el equipo de desarrollo
**Síntoma:** `failed to create a child event loop … Unable to establish loopback connection … UnixDomainSockets.connect0: Invalid argument`.
**Diagnóstico:** en Windows, Java 17 crea su *pipe* interno de NIO con un socket AF_UNIX; dentro de la terminal
restringida de la herramienta de desarrollo esas conexiones están bloqueadas (mismo origen que 7.1). Cambiar
`-Djdk.net.unixdomain.tmpdir` a una ruta corta no ayudó, lo que descartó el límite de longitud de ruta.
**Solución:** ejecutar `src/benchmark_spark.py` desde una terminal normal. Es otra razón práctica para no meter Spark en
la imagen: para este volumen añade una JVM y una superficie de fallos sin beneficio medible.

### 7.6 `log_loss` de scikit-learn con columnas desordenadas
**Síntoma:** la línea base "siempre local" aparecía con **mejor** log-loss que la casa de apuestas, algo imposible.
**Diagnóstico:** `sklearn.metrics.log_loss(y, p, labels=…)` asume que las columnas de `p` siguen el orden **alfabético**
de las etiquetas (Empate, Local, Visitante); el proyecto las ordena Local, Empate, Visitante.
**Solución:** log-loss explícito, `−mean(log p_clase_real)`, con el orden de columnas controlado (`src/modelo.py`).

### 7.7 Otros detalles resueltos
- **LightGBM 4.7:** `eval_set` está deprecado, pero la ruta nueva `eval_X/eval_y` no codifica etiquetas de texto
  (`Fields with bad pandas dtypes: resultado: str`) → se mantiene `eval_set` y se silencia solo ese aviso.
- **LightGBM en `python:3.13-slim`:** necesita `libgomp1` (OpenMP); sin él falla `OSError: libgomp.so.1`. Se instala en ambos Dockerfiles.
- **Modelos `.joblib`:** solo cargan con la misma versión de scikit-learn/LightGBM con que se entrenaron → versiones fijadas
  en ambos `requirements.txt`.
- **Streamlit 1.67:** la opción `theme.font` solo acepta **una** familia por URL de Google Fonts (`The source URL … contains multiple fonts`).

## 8. CRISP-DM en el proyecto

| Fase | Dónde |
|---|---|
| Business Understanding | Notebook §1 (pregunta, 5 preguntas específicas, tarea y criterios de éxito) |
| Data Understanding | Notebook §3 y §5 (diagnóstico de calidad, EDA, OLAP) |
| Data Preparation | `src/etl.py`, notebook §4 (4 técnicas con conteos antes/después, 2+ variables derivadas, Parquet) |
| Modeling | `src/modelo.py`, notebook §6 (anti-fuga, partición temporal, 2 líneas base, 3 algoritmos) |
| Evaluation | Notebook §7 (métricas, matriz de confusión, importancia, calibración, simulación de apuestas) |
| Deployment | MongoDB, PDF automático, dashboard y Docker |
