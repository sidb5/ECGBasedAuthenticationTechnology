from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import wfdb


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "04.GUI"

if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from feature_extraction import processing  # noqa: E402
from final_project import (  # noqa: E402
    Fiducial_Features,
    non_fiducial_features,
    non_fiducial_features_bonus_preprocessing,
)


SAMPLE_RECORDS = {
    "defined_104": {
        "expected": "subject 1",
        "path": REPO_ROOT / "06.Test" / "Defined_Signals" / "104" / "s0306lre",
    },
    "defined_117_a": {
        "expected": "subject 2",
        "path": REPO_ROOT / "06.Test" / "Defined_Signals" / "117" / "s0291lre",
    },
    "defined_117_b": {
        "expected": "subject 2",
        "path": REPO_ROOT / "06.Test" / "Defined_Signals" / "117" / "s0292lre",
    },
    "defined_122": {
        "expected": "subject 3",
        "path": REPO_ROOT / "06.Test" / "Defined_Signals" / "122" / "s0312lre",
    },
    "defined_234": {
        "expected": "subject 7",
        "path": REPO_ROOT / "06.Test" / "Defined_Signals" / "234" / "s0460_re",
    },
    "undefined_166": {
        "expected": "undefined",
        "path": REPO_ROOT / "06.Test" / "Undefined" / "166" / "s0275lre",
    },
    "undefined_238": {
        "expected": "undefined",
        "path": REPO_ROOT / "06.Test" / "Undefined" / "238" / "s0466_re",
    },
}


def load_models():
    return {
        "fiducial": pickle.load(open(GUI_DIR / "random_forest_classifier_Fid.pkl", "rb")),
        "non_fiducial": pickle.load(
            open(GUI_DIR / "random_forest_classifier_nonFid.pkl", "rb")
        ),
        "bonus_non_fiducial": pickle.load(
            open(GUI_DIR / "random_forest_classifier_nonFidBonus.pkl", "rb")
        ),
    }


def load_signal(record_path: Path):
    patient = wfdb.rdrecord(str(record_path), channels=[1])
    return processing(patient.p_signal[:, 0])


def summarize_prediction(probabilities: np.ndarray, threshold: float):
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])
    predicted_subject = top_index + 1
    decision = f"subject {predicted_subject}" if confidence >= threshold else "undefined"
    return {
        "decision": decision,
        "top_subject": predicted_subject,
        "confidence": confidence,
    }


def evaluate_record(models, processed_signal):
    results = {}

    _, fid_values = Fiducial_Features([processed_signal])
    fid_values = np.array(fid_values).reshape(-1, 23)
    fid_probabilities = models["fiducial"].predict_proba(fid_values[:, :22])[0]
    results["fiducial"] = summarize_prediction(fid_probabilities, threshold=0.8)

    _, non_fid_values = non_fiducial_features([processed_signal])
    non_fid_values = np.array(non_fid_values).reshape(1, 81)
    non_fid_probabilities = models["non_fiducial"].predict_proba(non_fid_values[:, :80])[0]
    results["non_fiducial"] = summarize_prediction(non_fid_probabilities, threshold=0.5)

    _, bonus_values = non_fiducial_features_bonus_preprocessing([processed_signal])
    bonus_array = np.array(bonus_values)
    if bonus_array.size == 0:
        results["bonus_non_fiducial"] = {
            "decision": "no_segments",
            "top_subject": None,
            "confidence": None,
        }
        return results

    bonus_values = bonus_array.reshape(-1, 41)
    bonus_probabilities = models["bonus_non_fiducial"].predict_proba(bonus_values[:, :40])[0]
    results["bonus_non_fiducial"] = summarize_prediction(
        bonus_probabilities, threshold=0.95
    )
    return results


def print_report(report):
    print("ECG Sample Evaluation")
    print("=" * 80)
    for sample_name, sample_result in report.items():
        print(f"{sample_name} | expected: {sample_result['expected']}")
        for method_name, method_result in sample_result["methods"].items():
            decision = method_result["decision"]
            top_subject = method_result["top_subject"]
            confidence = method_result["confidence"]
            confidence_text = (
                f"{confidence:.2%}" if isinstance(confidence, float) else "n/a"
            )
            print(
                f"  - {method_name}: decision={decision}, "
                f"top_subject={top_subject}, confidence={confidence_text}"
            )
        print("-" * 80)


def main():
    models = load_models()
    report = {}
    for sample_name, metadata in SAMPLE_RECORDS.items():
        processed_signal = load_signal(metadata["path"])
        report[sample_name] = {
            "expected": metadata["expected"],
            "methods": evaluate_record(models, processed_signal),
        }
    print_report(report)


if __name__ == "__main__":
    main()
