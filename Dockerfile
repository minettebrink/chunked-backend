# First stage
FROM python:3.11-slim as downloader


WORKDIR /app
RUN pip install --no-cache-dir huggingface_hub

COPY download_model.py .

RUN python3 download_model.py && \
    mkdir -p /app/model/chunks && \
    cd /root/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots/*/ && \
    split -b 2G ltx-video-2b-v0.9.5.safetensors /app/model/chunks/model.part

# Second stage
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model/chunks

COPY main.py .

COPY --from=downloader /app/model/chunks/model.partaa /app/model/chunks/
COPY --from=downloader /app/model/chunks/model.partab /app/model/chunks/
COPY --from=downloader /root/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots/*/model_index.json /app/model/
COPY --from=downloader /root/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots/*/ltx-video-2b-v0.9.5.license.txt /app/model/

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["bash", "-c", "cat /app/model/chunks/model.part* > /app/model/ltx-video-2b-v0.9.5.safetensors && rm -rf /app/model/chunks && uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug"]