# Confidence--Correctness Diagram

Reference implementation of the probability-mass decomposition and the class-specific **Confidence--Correctness Diagram** used in the manuscript *A Probabilistic Framework for Class-specific Evaluation of Diagnostic Systems*.

The repository separates the mathematical core, the class-specific diagram, and the experimental pipeline so that the diagram can be reused independently of the benchmark code.

## Repository structure

- `certainty_ratio.py` — classifier-independent mathematical core. It constructs `T`, `P`, `Q`, `Q_plus`, `Q_minus`, the hard and probabilistic confusion matrices, the decisive/residual matrices, and the corresponding scalar quantities.
- `confidence_correctness_diagram.py` — documented, reusable implementation of the global and class-specific `R/O/U/A` decomposition and the Confidence--Correctness Diagram. It uses `certainty_ratio.py` for the probability-mass mathematics.
- `confidence_correctness_pipeline.v5.py` — modularised version of pipeline.
- `examples/simple_diagram.py` — minimal stand-alone example using only true labels and probability vectors.

## Mathematical mapping

`certainty_ratio.py` denotes the decisive and residual probability-mass matrices by `V` and `U`. Pipeline used the names `H` and `L` for these same matrices. To preserve complete backward compatibility, `confidence_correctness_diagram.py` exports

```text
H = V = T.T @ Q_plus
L = U = T.T @ Q_minus
```
For each true class `alpha`, the row-normalised profile satisfies

```text
R_alpha + O_alpha + U_alpha + A_alpha = 1
```

where:

- `R` — Reliability: decisive mass assigned to the true class;
- `O` — Overconfidence: decisive mass assigned to a wrong class;
- `U` — Underconfidence: residual mass assigned to wrong classes when the top-1 prediction is correct;
- `A` — Ambiguity: residual mass retained by the true class when the top-1 prediction is wrong.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the versions used for the equivalence checks described below. The optional `imcp` dependency used by pipeline is not included in the supplied source files; MCP/IMCP columns are left as `NaN` when that module is unavailable.

## Stand-alone use

The diagram module can be used without the benchmark pipeline:

```python
import numpy as np

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

metrics, class_profiles, matrices = save_confidence_correctness_diagram(
    y_true=y_true,
    proba=proba,
    classes=classes,
    output_path="confidence_correctness.png",
    title="Example",
)

print(metrics)
print(class_profiles)
```

## Minimum script example (Iris dataset)
```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from confidence_correctness_diagram import save_confidence_correctness_diagram


iris = load_iris()
X = iris.data
y = iris.target_names[iris.target]
classes = iris.target_names

rf = RandomForestClassifier(n_estimators=500, random_state=42)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")

save_confidence_correctness_diagram(
    y_true=y,
    proba=proba,
    classes=classes,
    output_path="iris_confidence_correctness.png",
)
```
<img width="3269" height="908" alt="image" src="https://github.com/user-attachments/assets/53dd03de-6b58-41b7-b919-e9d304f29adc" />

## Reproducing the Leukaemia example (Figure 3)

Place the exact `Leukemia_GSE28497.csv` file used in the study in a directory by itself, for example:

```text
data/leukaemia/Leukemia_GSE28497.csv
```

The target column is expected to be `type`, as in the study datasets. To generate the raw Random Forest Confidence--Correctness Diagram using the same pipeline configuration:

```bash
python confidence_correctness_pipeline.v5.py \
  --data-dir data/leukaemia \
  --results-dir results \
  --images-dir Imagenes \
  --models RF \
  --folds 5 \
  --scaling standard \
  --no-calibration \
  --output-prefix figure3
```

The diagram is written to:

```text
Imagenes/figure3_Leukemia_GSE28497__RF__raw_confidence_correctness.png
```

Reproducing the exact published numerical values additionally requires the same dataset version used for the study.

## Notes on reproducibility

All model assessment should preferably use out-of-fold or otherwise genuinely held-out probability vectors. The stand-alone diagram functions do not fit a classifier and therefore do not enforce a validation design; that responsibility remains with the calling workflow.
