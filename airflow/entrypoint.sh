#!/bin/bash
set -e

# Function to wait for database
wait_for_db() {
    if [ -n "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN}" ]; then
        echo "Waiting for database..."
        timeout=60
        while ! airflow db check 2>/dev/null; do
            timeout=$((timeout - 1))
            if [ $timeout -le 0 ]; then
                echo "Database connection failed after 60 seconds"
                exit 1
            fi
            echo "Database not ready, waiting..."
            sleep 1
        done
        echo "Database is ready!"
    fi
}

# Initialize Airflow database
init_airflow() {
    echo "Initializing Airflow database..."
    airflow db migrate
    
    # Create admin user if it doesn't exist (using standalone method for Airflow 3.x)
    echo "Creating admin user..."
    airflow db-manager create-user \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin 2>/dev/null || echo "Admin user already exists or command not available"
    
    echo "Airflow initialization complete!"
}

case "$1" in
    webserver|api-server)
        wait_for_db
        init_airflow
        echo "Starting Airflow API server..."
        exec airflow api-server
        ;;
    scheduler)
        wait_for_db
        # Wait a bit to ensure webserver has initialized the DB
        sleep 10
        echo "Starting Airflow scheduler..."
        exec airflow scheduler
        ;;
    dag-processor)
        wait_for_db
        sleep 10
        echo "Starting Airflow DAG processor..."
        exec airflow dag-processor
        ;;
    worker|celery-worker)
        wait_for_db
        sleep 10
        echo "Starting Airflow Celery worker..."
        exec airflow celery worker
        ;;
    triggerer)
        wait_for_db
        sleep 10
        echo "Starting Airflow triggerer..."
        exec airflow triggerer
        ;;
    init)
        wait_for_db
        init_airflow
        ;;
    *)
        # If command is not recognized, execute it as-is
        exec "$@"
        ;;
esac
