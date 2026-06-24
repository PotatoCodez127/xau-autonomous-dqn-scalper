FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheelhouse .[dev]

FROM python:3.10-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheelhouse /wheelhouse

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheelhouse/* && \
    rm -rf /wheelhouse

COPY env_trading.py model_dqn.py train_dqn.py test_dqn.py pyproject.toml README.md ./

CMD ["python", "test_dqn.py"]