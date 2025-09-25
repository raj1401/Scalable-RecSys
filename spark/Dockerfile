FROM python:3.13-slim AS spark-base

ARG SPARK_VERSION=4.0.1

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

ARG SPARK_HOME="/opt/spark"
ENV SPARK_HOME=${SPARK_HOME}

ARG HADOOP_HOME="/opt/hadoop"
ENV HADOOP_HOME=${HADOOP_HOME}

ARG APP_HOME="/opt/app"

RUN mkdir -p ${HADOOP_HOME} && mkdir -p ${SPARK_HOME} && mkdir -p ${APP_HOME} && mkdir -p /opt/spark/history
WORKDIR ${SPARK_HOME}

# Install Spark
RUN curl https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz -o spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && tar xvzf spark-${SPARK_VERSION}-bin-hadoop3.tgz --directory /opt/spark --strip-components 1 \
    && rm -rf spark-${SPARK_VERSION}-bin-hadoop3.tgz

# Install UV package manager
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR ${APP_HOME}

# Prepare virtual environment using lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python /usr/local/bin/python

ENV VIRTUAL_ENV="${APP_HOME}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:/opt/spark/sbin:/opt/spark/bin:${PATH}"
ENV SPARK_MASTER="spark://spark-master:7077"
ENV SPARK_MASTER_HOST=spark-master
ENV SPARK_MASTER_PORT=7077
ENV SPARK_EVENTLOG_DIR="file:///opt/spark/history"
ENV SPARK_HISTORY_UI_PORT=18080
ENV SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=${SPARK_EVENTLOG_DIR} -Dspark.history.ui.port=${SPARK_HISTORY_UI_PORT}"
ENV SPARK_WORKER_DIR=/opt/spark/work
ENV SPARK_NO_DAEMONIZE=true
ENV PYSPARK_PYTHON="${VIRTUAL_ENV}/bin/python"

COPY spark/conf/spark-defaults.conf "$SPARK_HOME/conf"

RUN chmod u+x /opt/spark/sbin/* && \
    chmod u+x /opt/spark/bin/*

ENV PYTHONPATH="${SPARK_HOME}/python"

COPY spark/entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]