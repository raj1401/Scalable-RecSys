FROM apache/airflow:slim-latest-python3.13

USER root

# Install system dependencies needed for Spark provider
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script and ensure Unix line endings
COPY airflow/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

USER airflow

# Install Airflow providers and database drivers
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark \
    apache-airflow-providers-docker \
    psycopg2-binary \
    asyncpg \
    graphviz

# Set PYTHONPATH to include workspace (for DAG imports if needed)
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

WORKDIR /opt/airflow

EXPOSE 8081

ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]
