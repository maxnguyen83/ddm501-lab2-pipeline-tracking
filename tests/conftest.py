"""Fixtures shared by every test module."""

from pathlib import Path
from typing import Iterator

import mlflow
import pandas as pd
import pytest

from pipeline.config import TARGET
from pipeline.data_ingestion import load_raw


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """The real dataset, loaded once for the whole session."""
    return load_raw()


@pytest.fixture(scope="session")
def small(raw) -> pd.DataFrame:
    """A stratified 3,000-row sample — enough to fit, fast enough for CI."""
    return raw.groupby(TARGET, group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1500), random_state=0)
    ).reset_index(drop=True)


@pytest.fixture
def tracking(tmp_path: Path) -> Iterator[str]:
    """A throwaway MLflow file store and experiment; the previous URI is restored after."""
    from pipeline.training import setup_mlflow

    previous = mlflow.get_tracking_uri()
    uri = setup_mlflow(tracking_uri=f"file://{tmp_path / 'mlruns'}", experiment_name="test")
    yield uri
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(previous)
