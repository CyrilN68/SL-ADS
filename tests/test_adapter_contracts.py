"""tests/test_adapter_contracts.py

Contract tests for ``sl_ads.adapters``: every concrete adapter
(RedeRio, METR-LA, GECCO-IoT, CESNET-TimeSeries24) must inherit
``AdapterBase`` and produce a standardised output schema.

Phase H — added 2026-04-29.  Satisfies the ACM "Artifacts Evaluated —
Reusable" criterion *"artifacts are well-structured to facilitate
reuse and repurposing"* by guaranteeing the cross-dataset adapter
interface is a hard contract, not a convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.adapters.adapter_base import AdapterBase  # noqa: E402
from sl_ads.adapters.rederio_adapter import RederioAdapter  # noqa: E402
from sl_ads.adapters.gecco_adapter import GeccoAdapter  # noqa: E402
from sl_ads.adapters.cesnet_adapter import CesnetAdapter  # noqa: E402
from sl_ads.adapters.metr_la_adapter import MetrLaAdapter  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# 1. AdapterBase contract
# ────────────────────────────────────────────────────────────────────────
class TestAdapterBaseContract:
    def test_adapter_base_is_abstract(self):
        """``AdapterBase`` cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AdapterBase("RedeRio", {})  # type: ignore[abstract]

    @pytest.mark.parametrize("klass,name", [
        (RederioAdapter, "RederioAdapter"),
        (GeccoAdapter,   "GeccoAdapter"),
        (CesnetAdapter,  "CesnetAdapter"),
        (MetrLaAdapter,  "MetrLaAdapter"),
    ])
    def test_concrete_adapter_inherits_base(self, klass, name):
        assert issubclass(klass, AdapterBase), f"{name} must inherit AdapterBase"

    @pytest.mark.parametrize("klass", [
        RederioAdapter, GeccoAdapter, CesnetAdapter, MetrLaAdapter,
    ])
    def test_concrete_adapter_implements_required_methods(self, klass):
        """``load_raw_data`` and ``extract_metrics`` must be implemented."""
        assert "load_raw_data" in klass.__dict__ or any(
            "load_raw_data" in c.__dict__ for c in klass.__mro__[1:]
        )
        assert "extract_metrics" in klass.__dict__ or any(
            "extract_metrics" in c.__dict__ for c in klass.__mro__[1:]
        )


# ────────────────────────────────────────────────────────────────────────
# 2. Output schema invariants (RedeRio + GECCO — drivable from synth CSVs)
# ────────────────────────────────────────────────────────────────────────
class TestRederioOutputSchema:
    @pytest.fixture
    def rederio_synth_csv(self, tmp_path):
        csv = tmp_path / "rederio_synth.csv"
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="30s"),
            "bytes":   [100.0, 110.0, np.nan, 120.0, 115.0, 130.0, 125.0, 135.0, np.nan, 140.0],
            "packets": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "tcp":     [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        })
        df.to_csv(csv, index=False)
        return csv

    def test_rederio_extract_metrics_produces_required_columns(self, rederio_synth_csv):
        cfg = {"path_raw": str(rederio_synth_csv), "seasonality_period": 288}
        a = RederioAdapter("RedeRio", cfg)
        a.load_raw_data()
        a.extract_metrics()
        df = a.standardized_data
        assert df is not None
        assert "timestamp" in df.columns
        assert "label" in df.columns
        # timestamp must be datetime64
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        # rows preserved (no truncation)
        assert len(df) == 10

    def test_rederio_label_default_is_zero(self, rederio_synth_csv):
        cfg = {"path_raw": str(rederio_synth_csv), "seasonality_period": 288}
        a = RederioAdapter("RedeRio", cfg)
        a.load_raw_data()
        a.extract_metrics()
        # The synthetic CSV has no "label" column → adapter must default to 0.
        assert (a.standardized_data["label"] == 0).all()


class TestGeccoOutputSchema:
    @pytest.fixture
    def gecco_synth_dir(self, tmp_path):
        raw_dir = tmp_path / "gecco_raw"
        raw_dir.mkdir()
        for i, idx in enumerate(["01", "02"]):
            df = pd.DataFrame({
                "Time":   pd.date_range(f"2016-0{i+1}-01", periods=5, freq="1min"),
                "EVENT":  [False] * 4 + [True],
                "Tp":     [10.0, 11.0, 12.0, 13.0, 14.0],
                "Cl":     [0.5, 0.55, 0.5, 0.6, 0.5],
                "pH":     [7.0, 7.1, 7.05, 7.2, 7.0],
            })
            df.to_csv(raw_dir / f"{idx}_aug.csv", index=False)
        return raw_dir

    def test_gecco_extract_metrics_renames_columns(self, gecco_synth_dir):
        cfg = {"path_raw": str(gecco_synth_dir)}
        a = GeccoAdapter("GECCO-IoT", cfg)
        a.load_raw_data()
        a.extract_metrics()
        df = a.standardized_data
        assert df is not None
        # Time → timestamp, EVENT → label
        assert "timestamp" in df.columns
        assert "label" in df.columns
        assert "Time" not in df.columns
        assert "EVENT" not in df.columns
        # Concat-mode loaded both files (5 + 5 = 10 rows)
        assert len(df) == 10

    def test_gecco_label_is_int_01(self, gecco_synth_dir):
        cfg = {"path_raw": str(gecco_synth_dir)}
        a = GeccoAdapter("GECCO-IoT", cfg)
        a.load_raw_data()
        a.extract_metrics()
        df = a.standardized_data
        assert df["label"].dtype.kind in ("i", "u")  # integer
        # Each file had one True → 2 anomalies total
        assert int(df["label"].sum()) == 2


# ────────────────────────────────────────────────────────────────────────
# 3. NaN policy (audit_codex MAJ-01 enforcement at the contract level)
# ────────────────────────────────────────────────────────────────────────
class TestAdapterNaNPolicy:
    def test_rederio_preserves_nan_on_metrics(self, tmp_path):
        """``fillna(0)`` is forbidden on metric columns (audit_codex MAJ-01)."""
        csv = tmp_path / "rederio_nan.csv"
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="30s"),
            "bytes":     [100.0, np.nan, 200.0, np.nan, 300.0],
            "packets":   [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        df.to_csv(csv, index=False)
        cfg = {"path_raw": str(csv), "seasonality_period": 288}
        a = RederioAdapter("RedeRio", cfg)
        a.load_raw_data()
        a.extract_metrics()
        # NaNs preserved on each metric column
        assert a.standardized_data["bytes"].isna().sum() == 2
        assert a.standardized_data["packets"].isna().sum() == 1
