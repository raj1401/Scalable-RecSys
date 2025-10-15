FROM python:3.13-slim

ARG APP_HOME=/opt/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    curl \
    ca-certificates \
    vim \
    unzip \
    rsync \
    openjdk-21-jdk \
    build-essential \
    procps \
    ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR ${APP_HOME}

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python /usr/local/bin/python && \
    uv add --no-progress aiokafka fastapi "uvicorn[standard]" "pydantic>=2" && \
    uv sync --python /usr/local/bin/python

ENV VIRTUAL_ENV="${APP_HOME}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"
ENV PYTHONPATH="/workspace:${APP_HOME}"

CMD ["bash"]
