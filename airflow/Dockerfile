FROM apache/airflow:3.1.0-python3.13

USER root

# Install Java 21 and other system dependencies needed for Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    msopenjdk-21 \
    procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/msopenjdk-21-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Spark
ENV SPARK_VERSION=4.0.1
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"

RUN curl -fsSL https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    -o /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop3 ${SPARK_HOME} && \
    rm /tmp/spark.tgz && \
    chown -R airflow:root ${SPARK_HOME}

# Copy entrypoint script and ensure Unix line endings
COPY airflow/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

USER airflow

# Install Airflow providers and database drivers
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark \
    apache-airflow-providers-docker \
    apache-airflow-providers-celery>=3.3.0 \
    apache-airflow-providers-fab \
    flask-appbuilder \
    flask-session \
    connexion[flask] \
    psycopg2-binary \
    asyncpg \
    graphviz

# Set PYTHONPATH to include workspace (for DAG imports if needed)
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

WORKDIR /opt/airflow

EXPOSE 8081 8793

ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]
