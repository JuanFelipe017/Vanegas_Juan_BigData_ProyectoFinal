"""Configuración central del proyecto: rutas relativas, fuentes de datos y diccionarios de dominio.

Todas las rutas se calculan a partir de la raíz del proyecto (carpeta padre de ``src/``), así el mismo código
funciona en Windows, en Linux y dentro de los contenedores de Docker sin rutas absolutas.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------------------
RAIZ = Path(os.environ.get("PROYECTO_RAIZ", Path(__file__).resolve().parents[1]))
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
FIGURAS = REPORTS / "figuras"

for carpeta in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURAS):
    carpeta.mkdir(parents=True, exist_ok=True)

# Archivos de salida principales
PARQUET_PARTIDOS = DATA_PROCESSED / "partidos_limpios.parquet"
PARQUET_EQUIPOS = DATA_PROCESSED / "equipos_partido.parquet"        # formato largo (equipo × partido)
PARQUET_ESTADO = DATA_PROCESSED / "estado_equipos.parquet"          # último estado de cada equipo (para predecir)
PARQUET_ELO = DATA_PROCESSED / "elo_clubelo.parquet"
PARQUET_PREDICCIONES = DATA_PROCESSED / "predicciones_test.parquet"
PARQUET_MAPEO = DATA_PROCESSED / "mapeo_equipos_transfermarkt.parquet"
CUBO_LIGA_TEMPORADA = DATA_PROCESSED / "cubo_liga_temporada.parquet"
CUBO_PAIS_DECADA = DATA_PROCESSED / "cubo_pais_decada_nivel.parquet"
CUBO_EQUIPO_TEMPORADA = DATA_PROCESSED / "cubo_equipo_temporada.parquet"
JSON_ETL = DATA_PROCESSED / "resumen_etl.json"
JSON_MODELO = MODELS / "metricas_modelos.json"
MODELO_CON_CUOTAS = MODELS / "modelo_con_cuotas.joblib"
MODELO_SIN_CUOTAS = MODELS / "modelo_sin_cuotas.joblib"
REPORTE_PDF = REPORTS / "reporte_futbol_apuestas.pdf"
BANDERA_OK = REPORTS / ".pipeline_ok"

# --------------------------------------------------------------------------------------------------
# MongoDB (credenciales desde variables de entorno / archivo .env)
# --------------------------------------------------------------------------------------------------
# Fuera de Docker (kernel local) nadie carga el .env: se lee aquí sin pisar variables ya definidas
_ENV = RAIZ / ".env"
if _ENV.exists():
    for _linea in _ENV.read_text(encoding="utf-8").splitlines():
        _clave, _sep, _valor = _linea.partition("=")
        if _sep and not _linea.lstrip().startswith("#"):
            os.environ.setdefault(_clave.strip(), _valor.strip().strip('"').strip("'"))

MONGO_USER =os.environ.get("MONGO_USER", "admin")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "cambie_esta_clave")
MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_PORT = int(os.environ.get("MONGO_PORT", "27018"))
MONGO_DB = os.environ.get("MONGO_DB", "futbol_apuestas")
MONGO_COLECCION = "analisis"
MONGO_URI = os.environ.get(
    "MONGO_URI", f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
)

AUTOR = "Juan Felipe Vanegas Silva"
VIDEO_URL = os.environ.get("VIDEO_URL", "https://youtu.be/0VBYjifZxCE")

# --------------------------------------------------------------------------------------------------
# Fuentes de datos (todas públicas y verificables)
# --------------------------------------------------------------------------------------------------
URL_FD_MIRROR = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/main/data"
URL_TRANSFERMARKT = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

FUENTES = {
    "Matches.csv": {
        "url": f"{URL_FD_MIRROR}/Matches.csv",
        "fuente": "Football-Data.co.uk (resultados, estadísticas y cuotas) vía repositorio xgabora/Club-Football-Match-Data",
        "pagina": "https://www.football-data.co.uk/data.php",
    },
    "EloRatings.csv": {
        "url": f"{URL_FD_MIRROR}/EloRatings.csv",
        "fuente": "ClubElo (ratings Elo quincenales de clubes) vía repositorio xgabora/Club-Football-Match-Data",
        "pagina": "http://clubelo.com/",
    },
    "games.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/games.csv.gz",
        "fuente": "Transfermarkt (partidos con identificadores de club) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
    "clubs.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/clubs.csv.gz",
        "fuente": "Transfermarkt (clubes) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
    "competitions.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/competitions.csv.gz",
        "fuente": "Transfermarkt (competiciones) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
    "appearances.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/appearances.csv.gz",
        "fuente": "Transfermarkt (minutos de cada jugador en cada partido, todas las competiciones) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
    "players.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/players.csv.gz",
        "fuente": "Transfermarkt (jugadores: nacionalidad e internacionalidades) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
    "selecciones_results.csv": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "fuente": "Resultados de selecciones nacionales (eliminatorias, Nations League, amistosos) vía martj42/international_results",
        "pagina": "https://github.com/martj42/international_results",
    },
    "player_valuations.csv.gz": {
        "url": f"{URL_TRANSFERMARKT}/player_valuations.csv.gz",
        "fuente": "Transfermarkt (historial de valor de mercado de jugadores) vía dcaribou/transfermarkt-datasets",
        "pagina": "https://github.com/dcaribou/transfermarkt-datasets",
    },
}

# --------------------------------------------------------------------------------------------------
# Diccionario de ligas: código Football-Data → país, nivel, continente, nombre, código Transfermarkt
# Las ligas de "calendario anual" (temporada = año natural) se marcan con anual=True.
# --------------------------------------------------------------------------------------------------
LIGAS = {
    "E0": ("Inglaterra", 1, "Europa", "Premier League", "GB1", False),
    "E1": ("Inglaterra", 2, "Europa", "Championship", None, False),
    "E2": ("Inglaterra", 3, "Europa", "League One", None, False),
    "E3": ("Inglaterra", 4, "Europa", "League Two", None, False),
    "EC": ("Inglaterra", 5, "Europa", "National League", None, False),
    "SP1": ("España", 1, "Europa", "LaLiga", "ES1", False),
    "SP2": ("España", 2, "Europa", "LaLiga 2", None, False),
    "I1": ("Italia", 1, "Europa", "Serie A", "IT1", False),
    "I2": ("Italia", 2, "Europa", "Serie B", None, False),
    "D1": ("Alemania", 1, "Europa", "Bundesliga", "L1", False),
    "D2": ("Alemania", 2, "Europa", "2. Bundesliga", None, False),
    "F1": ("Francia", 1, "Europa", "Ligue 1", "FR1", False),
    "F2": ("Francia", 2, "Europa", "Ligue 2", None, False),
    "N1": ("Países Bajos", 1, "Europa", "Eredivisie", "NL1", False),
    "P1": ("Portugal", 1, "Europa", "Liga Portugal", "PO1", False),
    "B1": ("Bélgica", 1, "Europa", "Pro League", "BE1", False),
    "T1": ("Turquía", 1, "Europa", "Süper Lig", "TR1", False),
    "G1": ("Grecia", 1, "Europa", "Super League", "GR1", False),
    "SC0": ("Escocia", 1, "Europa", "Premiership", "SC1", False),
    "SC1": ("Escocia", 2, "Europa", "Championship (ESC)", None, False),
    "SC2": ("Escocia", 3, "Europa", "League One (ESC)", None, False),
    "SC3": ("Escocia", 4, "Europa", "League Two (ESC)", None, False),
    "AUT": ("Austria", 1, "Europa", "Bundesliga (AUT)", "A1", False),
    "SUI": ("Suiza", 1, "Europa", "Super League (SUI)", "C1", False),
    "DEN": ("Dinamarca", 1, "Europa", "Superliga", "DK1", False),
    "POL": ("Polonia", 1, "Europa", "Ekstraklasa", "PL1", False),
    "ROM": ("Rumania", 1, "Europa", "Liga I", "RO1", False),
    "RUS": ("Rusia", 1, "Europa", "Premier Liga", "RU1", False),
    "SWE": ("Suecia", 1, "Europa", "Allsvenskan", "SE1", True),
    "NOR": ("Noruega", 1, "Europa", "Eliteserien", "NO1", True),
    "FIN": ("Finlandia", 1, "Europa", "Veikkausliiga", None, True),
    "IRL": ("Irlanda", 1, "Europa", "Premier Division", None, True),
    "ARG": ("Argentina", 1, "América", "Liga Profesional", "ARG1", True),
    "BRA": ("Brasil", 1, "América", "Brasileirão", "BRA1", True),
    "MEX": ("México", 1, "América", "Liga MX", "MEX1", False),
    "USA": ("Estados Unidos", 1, "América", "MLS", "MLS1", True),
    "JAP": ("Japón", 1, "Asia", "J1 League", "JAP1", True),
    "CHN": ("China", 1, "Asia", "Super League (CHN)", None, True),
}

# Confederación de cada país de liga (para saber si un convocado cruzó de continente en el parón)
CONFEDERACION_PAIS = {"Argentina": "CONMEBOL", "Brasil": "CONMEBOL", "México": "CONCACAF", "Estados Unidos": "CONCACAF",
                      "Japón": "AFC", "China": "AFC"}          # el resto de países de liga son UEFA

# Nacionalidad de Transfermarkt → nombre de la selección en martj42/international_results
ALIAS_SELECCION = {
    "Türkiye": "Turkey", "Korea, South": "South Korea", "Korea, North": "North Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Cote d'Ivoire": "Ivory Coast", "Ireland": "Republic of Ireland",
    "Chinese Taipei": "Taiwan", "The Gambia": "Gambia", "Brunei Darussalam": "Brunei", "Curacao": "Curaçao",
    "Hongkong": "Hong Kong", "Southern Sudan": "South Sudan", "Sao Tome and Principe": "São Tomé and Príncipe",
    "St. Vincent & Grenadinen": "Saint Vincent and the Grenadines", "St. Lucia": "Saint Lucia",
}

RESULTADOS = {"H": "Local", "D": "Empate", "A": "Visitante"}
ORDEN_RESULTADOS = ["Local", "Empate", "Visitante"]

# Partición temporal para el modelo (sin barajar: se entrena con el pasado y se evalúa con el futuro)
FECHA_INICIO_VALIDACION = "2022-07-01"
FECHA_INICIO_TEST = "2023-07-01"
SEMILLA = 42
