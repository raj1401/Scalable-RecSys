# Airflow + Spark Docker Setup

## Overview

This setup provides a complete Apache Airflow environment integrated with your Spark cluster, all running in Docker containers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network (spark-net)               │
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │   PostgreSQL     │         │  Spark Master    │            │
│  │   (Database)     │         │  Port: 7077      │            │
│  │   Port: 5432     │         │  UI: 8080        │            │
│  └──────────────────┘         └────────┬─────────┘            │
│           │                             │                       │
│           │                    ┌────────┴────────┐             │
│           │                    │                 │             │
│           │              ┌─────▼─────┐    ┌─────▼─────┐       │
│           │              │  Spark    │    │  Spark    │       │
│           │              │  Worker 1 │    │  Worker 2 │       │
│           │              └───────────┘    └───────────┘       │
│           │                                                     │
│  ┌────────▼────────┐         ┌──────────────────┐            │
│  │  Airflow        │◄────────┤  Airflow         │            │
│  │  Webserver      │         │  Scheduler       │            │
│  │  Port: 8081     │         │                  │            │
│  └─────────────────┘         └──────────────────┘            │
│                                                                 │
│  Shared Volumes:                                               │
│  - ./airflow/dags     → /opt/airflow/dags                     │
│  - ./airflow/logs     → /opt/airflow/logs                     │
│  - .                  → /workspace                             │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### Spark Cluster
- **spark-master**: Spark master node (port 8080 for UI)
- **spark-worker**: Spark worker nodes (scalable)
- **spark-history**: Spark history server (port 18080)

### Airflow Stack
- **postgres**: PostgreSQL database for Airflow metadata
- **airflow-webserver**: Airflow web interface (port 8081)
- **airflow-scheduler**: Airflow scheduler (runs DAGs)

### Network
- **spark-net**: Bridge network connecting all services

## Quick Start

### 1. Build and Start All Services

```bash
# Build images
make build

# Start all services (Spark + Airflow)
make run-all
```

### 2. Access the Services

**Airflow Web UI:**
- URL: http://localhost:8081
- Username: `admin`
- Password: `admin`

**Spark Master UI:**
- URL: http://localhost:8080

**Spark History Server:**
- URL: http://localhost:18080

### 3. Verify Airflow Connection

1. Open Airflow UI (http://localhost:8081)
2. Go to **Admin → Connections**
3. Find `spark_default` connection
4. Should show: `spark://spark-master:7077`

### 4. Test Your DAG

```bash
# List all DAGs
make airflow-list-dags

# Trigger the main DAG
make airflow-trigger-dag dag=etl_train_test_pipeline

# Watch logs
make airflow-logs
```

## Makefile Commands

### General Commands

```bash
make build              # Build all Docker images
make build-nc           # Build without cache
make down               # Stop and remove all containers
make run-all            # Start all services (Spark + Airflow)
make stop               # Stop all containers
```

### Service-Specific Commands

```bash
make run-spark-only     # Start only Spark cluster
make run-airflow-only   # Start only Airflow (+ Postgres)
make logs-all           # View all logs
```

### Airflow Commands

```bash
make airflow-init               # Initialize Airflow (one-time)
make airflow-logs               # View Airflow logs
make airflow-webserver-logs     # View webserver logs only
make airflow-scheduler-logs     # View scheduler logs only
make airflow-list-dags          # List all DAGs
make airflow-trigger-dag dag=<dag_id>  # Trigger a DAG
make airflow-bash               # Open bash in Airflow container
make airflow-clean              # Clean Airflow data and volumes
```

### Spark Commands (still available)

```bash
make submit-standalone-spark-etl
make submit-standalone-spark-train
make submit-standalone-spark-test
make submit-standalone-spark-future
```

## Environment Variables

### Airflow Configuration

Set in `docker-compose.yml` under Airflow services:

```yaml
# Database
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow

# Spark Connection
AIRFLOW_CONN_SPARK_DEFAULT: spark://spark-master:7077

# Core Settings
AIRFLOW__CORE__EXECUTOR: LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES: 'False'
```

### Customizing Spark Connection

To modify Spark connection settings:

1. **Via Environment Variable** (recommended):
   ```yaml
   AIRFLOW_CONN_SPARK_DEFAULT: spark://spark-master:7077?deploy-mode=client
   ```

2. **Via Airflow UI**:
   - Admin → Connections → Edit `spark_default`

## Volume Mounts

### Airflow Volumes

```yaml
- ./airflow/dags:/opt/airflow/dags        # DAG files
- ./airflow/logs:/opt/airflow/logs        # Airflow logs
- ./airflow/plugins:/opt/airflow/plugins  # Custom plugins
- ./airflow/config:/opt/airflow/config    # Configuration
- .:/workspace                            # Full workspace (for Spark apps)
```

### Spark Volumes

```yaml
- .:/workspace                            # Full workspace
- spark-history:/opt/spark/history        # Spark history logs
```

## Port Mappings

| Service | Container Port | Host Port | Description |
|---------|---------------|-----------|-------------|
| Spark Master UI | 8080 | 8080 | Spark cluster UI |
| Spark Master | 7077 | 7077 | Spark master |
| Spark History | 18080 | 18080 | Spark history server |
| Airflow Webserver | 8081 | 8081 | Airflow UI |
| PostgreSQL | 5432 | 5432 | Database |

## First-Time Setup

### Complete Setup Process

```bash
# 1. Build images
make build

# 2. Start all services
make run-all

# 3. Wait for services to start (30-60 seconds)
# Watch logs to see when ready:
make airflow-logs

# 4. Access Airflow UI
# http://localhost:8081
# Login: admin / admin

# 5. Verify DAG appears
make airflow-list-dags

# 6. Trigger the DAG
make airflow-trigger-dag dag=etl_train_test_pipeline
```

## Troubleshooting

### Issue: Airflow UI not accessible

**Check if containers are running:**
```bash
docker ps | grep airflow
```

**Check logs:**
```bash
make airflow-webserver-logs
```

**Solution:**
- Wait 30-60 seconds for initialization
- Check that postgres is healthy: `docker ps | grep postgres`

### Issue: DAG not appearing

**Verify DAG file location:**
```bash
docker compose exec airflow-webserver ls -la /opt/airflow/dags/
```

**Check for syntax errors:**
```bash
docker compose exec airflow-webserver airflow dags list-import-errors
```

**Solution:**
- Ensure DAG file is in `airflow/dags/` directory
- Wait 30 seconds for scheduler to pick up changes
- Check logs for Python errors

### Issue: Spark connection fails

**Test Spark connectivity:**
```bash
docker compose exec airflow-webserver ping spark-master
```

**Check Spark master:**
```bash
docker ps | grep spark-master
```

**Verify connection in Airflow:**
```bash
docker compose exec airflow-webserver airflow connections get spark_default
```

**Solution:**
- Ensure Spark cluster is running
- Verify both on same network (`spark-net`)
- Check environment variable `AIRFLOW_CONN_SPARK_DEFAULT`

### Issue: Database connection errors

**Check PostgreSQL:**
```bash
docker compose exec postgres pg_isready -U airflow
```

**Reinitialize database:**
```bash
make airflow-clean
make run-all
```

### Issue: Permission errors

**Fix volume permissions:**
```bash
sudo chown -R $USER:$USER airflow/logs airflow/config
```

## Development Workflow

### 1. Adding a New DAG

1. Create DAG file in `airflow/dags/`
2. Wait 30 seconds for scheduler to detect
3. Check in Airflow UI or run: `make airflow-list-dags`
4. Trigger manually or wait for schedule

### 2. Testing DAGs

```bash
# Test DAG structure
make airflow-test-dag dag=etl_train_test_pipeline

# Trigger DAG
make airflow-trigger-dag dag=etl_train_test_pipeline

# Watch logs
make airflow-logs
```

### 3. Debugging

```bash
# Access Airflow container
make airflow-bash

# Inside container:
airflow dags list
airflow dags list-import-errors
airflow tasks list etl_train_test_pipeline
python /opt/airflow/dags/dag_etl_train.py  # Test import
```

### 4. Updating Configuration

Edit `docker-compose.yml` → Restart services:
```bash
docker compose up -d airflow-webserver airflow-scheduler
```

## Production Considerations

### Scaling

**Scale Spark workers:**
```bash
docker compose up -d --scale spark-worker=5
```

**Switch to Celery Executor** (for production):
1. Add Redis/RabbitMQ service
2. Change executor in environment:
   ```yaml
   AIRFLOW__CORE__EXECUTOR: CeleryExecutor
   ```
3. Add worker service

### Security

**Change default password:**
```bash
make airflow-bash
airflow users create \
    --username your_user \
    --firstname Your \
    --lastname Name \
    --role Admin \
    --email your@email.com \
    --password your_password
```

**Use secrets for production:**
- Don't commit passwords to git
- Use Docker secrets or environment files
- Enable RBAC and authentication

### Monitoring

**View service status:**
```bash
docker compose ps
```

**Resource usage:**
```bash
docker stats
```

**Airflow task logs:**
- Available in Airflow UI under task instance
- Also in `airflow/logs/` directory

## File Structure

```
.
├── airflow/
│   ├── Dockerfile                 # Airflow Docker image
│   ├── entrypoint.sh              # Airflow entrypoint script
│   ├── dags/                      # DAG files
│   │   ├── dag_etl_train.py       # Main DAG
│   │   ├── common/                # Shared utilities
│   │   └── examples/              # Example DAGs
│   ├── logs/                      # Airflow logs (gitignored)
│   ├── config/                    # Airflow config (gitignored)
│   └── plugins/                   # Custom plugins
├── spark/
│   ├── Dockerfile                 # Spark Docker image
│   ├── apps/                      # Spark applications
│   └── conf/                      # Spark configuration
├── docker-compose.yml             # Docker Compose configuration
└── Makefile                       # Command shortcuts
```

## Cleanup

### Remove all containers and volumes

```bash
make down
```

### Clean Airflow data

```bash
make airflow-clean
```

### Complete cleanup

```bash
make down
rm -rf airflow/logs/* airflow/config/*
docker system prune -a
```

## Next Steps

1. ✅ Start services: `make run-all`
2. ✅ Access Airflow UI: http://localhost:8081
3. ✅ Verify DAG: `make airflow-list-dags`
4. ✅ Trigger DAG: `make airflow-trigger-dag dag=etl_train_test_pipeline`
5. ✅ Monitor execution in Airflow UI
6. ✅ Check Spark UI for job details: http://localhost:8080

## Resources

- **Airflow Docs**: https://airflow.apache.org/docs/
- **Spark Docs**: https://spark.apache.org/docs/latest/
- **Docker Compose**: https://docs.docker.com/compose/

## Support

For issues or questions:
1. Check logs: `make airflow-logs`
2. Review troubleshooting section above
3. Check DAG import errors: `airflow dags list-import-errors`
4. Inspect container: `make airflow-bash`
