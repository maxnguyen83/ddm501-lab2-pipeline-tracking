# =============================================================================
# Airflow image with the pipeline's dependencies
# DDM501 - Lab 2
#
# The stock apache/airflow image cannot import the `pipeline` package (no
# scikit-learn, no mlflow). This image adds them, installed under Airflow's
# own constraints so the scheduler keeps the versions it was released with.
# =============================================================================
FROM apache/airflow:2.8.4-python3.11

ARG AIRFLOW_VERSION=2.8.4

# -----------------------------------------------------------------------------
# 1. Switch to the airflow user before installing
# -----------------------------------------------------------------------------
# The base image starts as root. pip install as root inside an Airflow image
# writes to the wrong site-packages and the scheduler will not see the packages.
USER airflow

# docker-compose mounts ./pipeline and ./data under /opt/airflow/project.
# Creating the directory here, as airflow, means it is not root-owned when the
# mounts appear, so pipeline/config.py can create models/ and artifacts/ in it.
# g+rwX because compose may run the container as AIRFLOW_UID with group 0.
RUN mkdir -p /opt/airflow/project /opt/airflow/artifacts \
    && chmod -R g+rwX /opt/airflow/project /opt/airflow/artifacts

# -----------------------------------------------------------------------------
# 2. Install requirements-airflow.txt WITH Airflow's constraints file
# -----------------------------------------------------------------------------
# The constraints file is per Python minor version. Reading the version from the
# interpreter (instead of hard-coding 3.11) keeps the URL right if the base tag
# changes. numpy/pandas/pyarrow are decided by the constraints; scikit-learn,
# joblib and mlflow are not in it and stay pinned to match requirements.txt.
COPY --chown=airflow:root requirements-airflow.txt /tmp/requirements-airflow.txt
RUN PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    && pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        -r /tmp/requirements-airflow.txt \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
    && pip check

# -----------------------------------------------------------------------------
# 3. Set PYTHONPATH so the DAG can import `pipeline`
# -----------------------------------------------------------------------------
# docker-compose.yml mounts ./pipeline to /opt/airflow/project/pipeline.
ENV PROJECT_ROOT=/opt/airflow/project \
    PYTHONPATH=/opt/airflow/project
