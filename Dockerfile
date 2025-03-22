FROM python:3.12.4-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3

RUN apt-get update && apt-get install -y --no-install-recommends iputils-ping net-tools \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir .

CMD ["sleep", "infinity"]
