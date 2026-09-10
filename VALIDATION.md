# v4--v5 validation record

The supplied `confidence_correctness_pipeline.v4.py` was compared with the modularised `confidence_correctness_pipeline.v5.py` using the same synthetic three-class dataset, software environment and random seeds.

Validated outputs:

- `detailed_results.csv`: exact DataFrame equality;
- `class_specific_results.csv`: exact DataFrame equality;
- `reliability_bins.csv`: exact DataFrame equality;
- `aggregate_summary_by_model.csv`: exact DataFrame equality;
- generated reliability PNGs: byte-identical;
- generated Confidence--Correctness PNGs: byte-identical;
- generated Excel workbooks: identical worksheet names and cell values.

Two checks were run:

1. raw RF and LR, 3-fold outer stratified CV;
2. RF raw and sigmoid-calibrated, 3-fold outer stratified CV and 2-fold inner calibration.

Validation environment:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- Matplotlib 3.10.8
- scikit-learn 1.8.0
- openpyxl 3.1.5

The test establishes equivalence of the refactoring in the tested conditions. As usual for stochastic or version-sensitive machine-learning software, reproducing published values also requires the same data and compatible library versions.
