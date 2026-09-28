# SonicSentinel AI as a Hugging Face Docker Space (CPU basic: 2 vCPU, 16 GB RAM).
# Built from the folder scripts/deploy_hf_space.sh assembles from the committed tree;
# see deploy/huggingface/SPACE.md. Also builds locally:
#   docker build -f deploy/huggingface/Dockerfile -t sonicsentinel .
#   docker run -p 7860:7860 -e SST_SECRET_KEY=... sonicsentinel
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; everything it writes lives under its home.
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    SST_DB_PATH=/home/app/data/sonicsentinel.db \
    SST_STORAGE_DIR=/home/app/data/storage \
    SST_PRODUCTION=1 \
    SST_LOAD_MODELS=1 \
    SST_TORCH_THREADS=2 \
    SST_TRUSTED_PROXY_HOPS=1
WORKDIR /home/app/sonicsentinel

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app
# The CNN14, AST and CLAP weights go into the image so a restart never waits on a download.
RUN python tools/fetch_pretrained.py --all

EXPOSE 7860
CMD ["sh", "deploy/huggingface/start.sh"]
