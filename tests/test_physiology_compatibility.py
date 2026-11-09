import json
from pathlib import Path

import numpy as np
import pytest

from multisensor_ml.physiology_compatibility import (
    consume_descriptors,
    eeg_descriptors,
    load_schema,
    rr_rmssd,
    watch_core,
)

ROOT = Path(__file__).resolve().parents[1] / "contracts/physiology/v1"


@pytest.mark.parametrize("source,code,value,accepted", [
    ("watch", "hr_median_bpm", 0, False),
    ("h10", "rr_median_ms", -1, False),
    ("watch", "eda_median_us", -0.01, False),
    ("h10", "rr_rmssd_ms", -0.01, False),
    ("h10", "ecg_ac_rms_uv", -0.01, False),
    ("muse", "eeg_1_alpha_relative_power", 1.00001, False),
    ("muse", "eeg_1_spectral_entropy", -0.01, False),
    ("muse", "eeg_1_spectral_entropy", 0, True),
    ("muse", "eeg_1_alpha_relative_power", 1, True),
    ("watch", "eda_median_us", 0, True),
    ("h10", "rr_rmssd_ms", 0, True),
    ("watch", "eda_slope_us_per_s", -1, True),
    ("watch", "skin_temp_median_c", -1, True),
])
def test_descriptor_domains(source, code, value, accepted):
    filename = {"watch": "watch_physio_core_v1", "h10": "h10_physio_core_v1",
                "muse": "muse_eeg_core_v1"}[source]
    schema = load_schema(json.loads((ROOT / f"{filename}.json").read_text()), source)
    rows = [dict(code=d["code"], unit=d["unit"], value=value if d["code"] == code else 1.,
                 readiness="AVAILABLE", reason=None) for d in schema["descriptors"]]
    if accepted:
        assert consume_descriptors(schema, rows) is not None
    else:
        with pytest.raises(ValueError, match="domain"):
            consume_descriptors(schema, rows)


@pytest.mark.parametrize("source,filename,width", [
    ("watch", "watch_physio_core_v1.json", 3),
    ("h10", "h10_physio_core_v1.json", 3),
    ("muse", "muse_eeg_core_v1.json", 24),
])
def test_pinned_schema_and_optional_absence(source, filename, width):
    doc = json.loads((ROOT / filename).read_text())
    schema = load_schema(doc, source)
    previous_draft = json.loads(json.dumps(doc))
    for descriptor in previous_draft["descriptors"]:
        descriptor.pop("domain")
    with pytest.raises(ValueError, match="hash"):
        load_schema(previous_draft, source)
    rows = [{"code": d["code"], "unit": d["unit"], "value": 1.0,
             "readiness": "AVAILABLE", "reason": None}
            for d in doc["descriptors"] if d["required"]]
    assert consume_descriptors(schema, rows) == (1.0,) * width
    rows[0]["unit"] = "wrong"
    with pytest.raises(ValueError):
        consume_descriptors(schema, rows)
    doc["policy"]["wave_min_coverage"] = 1
    with pytest.raises(ValueError, match="hash"):
        load_schema(doc, source)
    doc["policy"]["wave_min_coverage"] = 1.0
    doc["core_features"].reverse()
    with pytest.raises(ValueError, match="hash"):
        load_schema(doc, source)


def test_core_null_blocks_and_extra_fields_rejected():
    schema = load_schema(json.loads((ROOT / "h10_physio_core_v1.json").read_text()), "h10")
    assert consume_descriptors(schema, []) is None
    with pytest.raises(ValueError):
        consume_descriptors(schema, [{"code": "motion", "unit": "g", "value": 1.0,
                                     "readiness": "AVAILABLE", "reason": None}])


@pytest.mark.parametrize("mutation", ["nan", "private", "duplicate", "null", "unknown"])
def test_invalid_descriptor_rows_fail_closed(mutation):
    schema = load_schema(json.loads((ROOT / "h10_physio_core_v1.json").read_text()), "h10")
    row = dict(code="hr_median_bpm", unit="bpm", value=72.0,
               readiness="AVAILABLE", reason=None)
    rows = [row]
    if mutation == "nan":
        row["value"] = float("nan")
    elif mutation == "private":
        row["person_uuid"] = "private"
    elif mutation == "duplicate":
        rows.append(dict(row))
    elif mutation == "null":
        row["value"] = None
    else:
        row["readiness"] = "UNKNOWN"
    with pytest.raises(ValueError):
        consume_descriptors(schema, rows)


def test_watch_literal_and_rr_boundaries():
    assert watch_core(list(range(60, 70)), list(range(1, 11)), list(range(1, 11))) == (
        64.5, 5.5, 1.0)
    assert rr_rmssd([800, 810, 800, 820], [True]*4, [0]*4, [0,1,2,3]) == pytest.approx(
        np.sqrt(200))
    assert rr_rmssd([800,810,1000,1040], [True]*4, [0]*4, [0,1,3,4]) == pytest.approx(
        np.sqrt(850))
    assert rr_rmssd([800,810,1000,1040], [True,False,True,True], [0,0,1,1],
                    [0,1,2,3]) == 40.0
    assert rr_rmssd([800,810], [True]*2, [0,1], [0,1]) is None


def test_muse_known_sine_and_invalid_signal():
    x = 2*np.sin(2*np.pi*10*np.arange(2560)/256)
    vector = eeg_descriptors(x, 256)
    assert vector[:5] == pytest.approx([0, 0, 1, 0, 0], abs=1e-12)
    expected_entropy = -(2/3*np.log(2/3) + 2*(1/6*np.log(1/6)))/np.log(176)
    assert vector[5] == pytest.approx(expected_entropy, abs=1e-12)
    assert eeg_descriptors(7*x+13, 256) == pytest.approx(vector, abs=1e-12)
    for bad in [x[:-1], np.zeros(2560), np.full(2560, np.nan)]:
        with pytest.raises(ValueError):
            eeg_descriptors(bad, 256)
