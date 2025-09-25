#!/bin/bash
set -euo pipefail

SPARK_WORKLOAD=${1:-"worker"}
echo "Starting Spark workload: ${SPARK_WORKLOAD}"

export SPARK_NO_DAEMONIZE=${SPARK_NO_DAEMONIZE:-true}
export SPARK_MASTER_URL=${SPARK_MASTER:-"spark://spark-master:7077"}
export SPARK_HISTORY_OPTS=${SPARK_HISTORY_OPTS:-"-Dspark.history.fs.logDirectory=file:///opt/spark/history -Dspark.history.ui.port=18080"}

case "${SPARK_WORKLOAD}" in
  master)
    exec start-master.sh --host "${SPARK_MASTER_HOST:-spark-master}" --port "${SPARK_MASTER_PORT:-7077}"
    ;;
  worker)
    exec start-worker.sh "${SPARK_MASTER_URL}"
    ;;
  history)
    mkdir -p /opt/spark/history
    exec start-history-server.sh
    ;;
  *)
    echo "Unknown workload: ${SPARK_WORKLOAD}"
    exec "$@"
    ;;
esac