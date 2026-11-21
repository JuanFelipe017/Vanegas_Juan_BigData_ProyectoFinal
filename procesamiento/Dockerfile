# Servicio de procesamiento: ejecuta el pipeline (descarga → ETL → ML → MongoDB → PDF) y deja JupyterLab abierto
FROM python:3.13-slim

# libgomp1: OpenMP que necesita LightGBM en Linux (sin él: "OSError: libgomp.so.1: cannot open shared object file")
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg PROYECTO_RAIZ=/proyecto \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /proyecto

COPY procesamiento/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# El código va en la imagen; data/, models/, reports/ y notebooks/ llegan como volúmenes (persisten en el equipo)
COPY src/ /proyecto/src/

EXPOSE 8888
# 1) pipeline idempotente (no re-entrena si los modelos ya existen) 2) JupyterLab para revisar/ejecutar el notebook
CMD ["sh", "-c", "python src/pipeline.py && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/proyecto --IdentityProvider.token=\"${JUPYTER_TOKEN:-bigdata}\""]
