# Confidence--Correctness Diagram

Reference implementation of the probability-mass decomposition and the class-specific **Confidence--Correctness Diagram** used in the manuscript *A Probabilistic Framework for Class-specific Evaluation of Diagnostic Systems*.

The repository separates the mathematical core, the class-specific diagram, and the experimental pipeline so that the diagram can be reused independently of the benchmark code.

## Repository structure

- `certainty_ratio.py` — classifier-independent mathematical core. It constructs `T`, `P`, `Q`, `Q_plus`, `Q_minus`, the hard and probabilistic confusion matrices, the decisive/residual matrices, and the corresponding scalar quantities.
- `confidence_correctness_diagram.py` — documented, reusable implementation of the global and class-specific `R/O/U/A` decomposition and the Confidence--Correctness Diagram. It uses `certainty_ratio.py` for the probability-mass mathematics.
- `confidence_correctness_pipeline.v5.py` — modularised version of pipeline v4. It preserves the experimental design and output schema of v4, but delegates the mathematical decomposition to `certainty_ratio.py` and the class-specific analysis/plotting to `confidence_correctness_diagram.py`.
- `examples/simple_diagram.py` — minimal stand-alone example using only true labels and probability vectors.
- `tests/test_confidence_correctness_diagram.py` — numerical and structural tests for the class-specific decomposition.

## Mathematical mapping

`certainty_ratio.py` denotes the decisive and residual probability-mass matrices by `V` and `U`. Pipeline v4 used the names `H` and `L` for these same matrices. To preserve complete backward compatibility, `confidence_correctness_diagram.py` exports

```text
H = V = T.T @ Q_plus
L = U = T.T @ Q_minus
```

The symbol `U` in the class-specific `R/O/U/A` profile denotes **underconfidence** and must not be confused with the residual matrix `U` used internally by `certainty_ratio.py`.

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

The versions in `requirements.txt` are the versions used for the equivalence checks described below. The optional `imcp` dependency used by pipeline v4 is not included in the supplied source files; as in v4, MCP/IMCP columns are left as `NaN` when that module is unavailable.

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

The convenience function derives hard predictions from the largest probability when `y_pred` is not supplied. When the module is used from pipeline v5, the classifier's hard predictions are supplied explicitly so that the hard confusion matrix remains identical to pipeline v4.

## Reproducing the Leukaemia example (Figure 3)

Place the exact `Leukemia_GSE28497.csv` file used in the study in a directory by itself, for example:

```text
data/leukaemia/Leukemia_GSE28497.csv
```

The target column is expected to be `type`, as in the study datasets. To generate the raw Random Forest Confidence--Correctness Diagram using the same pipeline configuration as v4:

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

Reproducing the exact published numerical values additionally requires the same dataset version and software environment used for the study.

## Full experimental pipeline

Running v5 without a model subset evaluates the same classifiers defined in v4 (`RF`, `LR`, and `MLP`) over every CSV file in the data directory:

```bash
python confidence_correctness_pipeline.v5.py \
  --data-dir ../data_nature \
  --results-dir results \
  --images-dir Imagenes
```

The command-line interface, classifier definitions, cross-validation procedure, calibration options, output tables, filenames, matrix exports, reliability diagrams, and Confidence--Correctness diagrams are preserved from v4.

## v4 / v5 equivalence

The modularisation was checked against the supplied `confidence_correctness_pipeline.v4.py` on a fixed synthetic three-class dataset. With the same environment and random seed:

- detailed CSV results were exactly equal;
- class-specific CSV results were exactly equal;
- reliability-bin and aggregate CSV results were exactly equal;
- all generated PNG files were byte-identical;
- all worksheet names and cell values in the generated Excel workbooks were identical.

The comparison was performed for raw RF/LR runs and for raw plus sigmoid-calibrated RF runs. Thus, the refactoring changes code organisation, not the numerical definitions or experimental results.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

A quick end-to-end pipeline self-test is also available:

```bash
python confidence_correctness_pipeline.v5.py --self-test
```

## Notes on reproducibility

All model assessment should preferably use out-of-fold or otherwise genuinely held-out probability vectors. The stand-alone diagram functions do not fit a classifier and therefore do not enforce a validation design; that responsibility remains with the calling workflow.
