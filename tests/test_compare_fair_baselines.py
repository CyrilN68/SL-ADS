import numpy as np
import pandas as pd

from sl_ads.compare.compare_no_sl_fair import (
    _align_evidence_to_detection,
    _calibrate_threshold as _calibrate_no_sl_threshold,
)
from sl_ads.compare.compare_raw_baselines_fair import (
    _calibrate_threshold as _calibrate_raw_threshold,
)


def test_no_sl_alignment_uses_detection_timeline_with_first_window_offset():
    detection = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-11-10 00:00:00", "2025-11-10 00:05:00"]),
            "score": [0.1, 0.2],
        }
    )
    evidence = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-11-10 00:04:30", "2025-11-10 00:05:30"]),
            "leaf_N": [4.0, 7.0],
            "leaf_P": [3.0, 1.0],
            "leaf_S": [3.0, 2.0],
        }
    )

    aligned = _align_evidence_to_detection(evidence, detection, ["leaf"])

    assert aligned["timestamp"].tolist() == detection["timestamp"].tolist()
    assert aligned["leaf_N"].tolist() == [4.0, 7.0]


def test_no_sl_threshold_calibration_never_exceeds_target_fpr():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    threshold, empirical = _calibrate_no_sl_threshold(scores, target_fpr=0.21)

    assert threshold == 0.9
    assert empirical <= 0.21


def test_raw_threshold_calibration_never_exceeds_target_fpr():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    threshold, empirical = _calibrate_raw_threshold(scores, target_fpr=0.26)

    assert threshold == 4.0
    assert empirical <= 0.26


def test_raw_threshold_calibration_handles_tied_max_scores():
    scores = np.array([1.0, 2.0, 9.0, 9.0, 9.0])
    threshold, empirical = _calibrate_raw_threshold(scores, target_fpr=0.2)

    assert threshold > 9.0
    assert empirical == 0.0
