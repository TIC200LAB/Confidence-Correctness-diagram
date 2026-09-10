#!/usr/bin/env python3
"""Minimal stand-alone Confidence--Correctness Diagram example."""

from pathlib import Path
import sys

import numpy as np

# Allow the example to be run directly from the repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_correctness_diagram import save_confidence_correctness_diagram


classes = np.array(["A", "B", "C"])
y_true = np.array(["A", "A", "B", "B", "C", "C"])
proba = np.array([
    [0.80, 0.10, 0.10],
    [0.30, 0.60, 0.10],
    [0.10, 0.80, 0.10],
    [0.20, 0.70, 0.10],
    [0.10, 0.20, 0.70],
    [0.60, 0.20, 0.20],
])

output = REPO_ROOT / "examples" / "simple_confidence_correctness.png"
metrics, class_profiles, _ = save_confidence_correctness_diagram(
    y_true=y_true,
    proba=proba,
    classes=classes,
    output_path=output,
    title="Simple Confidence--Correctness example",
)

print("Global decomposition:")
for key in ("R", "O", "U", "A", "acc_star", "lambda_h", "lambda_l"):
    print(f"  {key}: {metrics[key]:.6f}")

print("\nClass-specific profiles:")
print(
    class_profiles[
        [
            "class",
            "n_class",
            "R_alpha_row_norm",
            "O_alpha_row_norm",
            "U_alpha_row_norm",
            "A_alpha_row_norm",
        ]
    ].to_string(index=False)
)
print(f"\nDiagram saved to: {output}")
