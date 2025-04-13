FROM python:3.12.4-slim

ENV TF_CPP_MIN_LOG_LEVEL=3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    net-tools iproute2 iputils-ping && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/netfl
COPY netfl .
COPY pyproject.toml .
RUN pip install --no-cache-dir .

WORKDIR /app
COPY run.py .
