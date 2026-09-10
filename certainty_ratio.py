"""Core probability-mass evaluation objects defined in the paper.

The module converts instance-level class probabilities into the matrices
introduced in the manuscript:

    CM      = T.T @ P
    CM_star = T.T @ Q
    Q       = Q_plus + Q_minus
    CM_star = V + U
    V       = T.T @ Q_plus
    U       = T.T @ Q_minus

It also computes the global decisiveness and residual-dispersion weights,
``lambda_v`` and ``lambda_u``, together with hard accuracy ``Acc`` and
probabilistic accuracy ``Acc_star``.

Only NumPy is required. The implementation is independent of the classifier
that generated the probability matrix ``Q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int64]


@dataclass(frozen=True)
class ProbabilityMassEvaluation:
    """Container for all matrices and scalar measures in the framework.

    Attributes
    ----------
    classes:
        Class labels in the same order as the columns of ``Q``.
    y_index:
        Integer-encoded true labels, in ``0, ..., k-1``.
    predicted_index:
        Tie-resolved argmax predictions, in ``0, ..., k-1``.
    T:
        One-hot ground-truth matrix.
    P:
        One-hot hard-prediction matrix.
    Q:
        Validated class-probability matrix supplied by the caller.
    Q_plus:
        Matrix retaining only the selected top-1 probability in each row.
    Q_minus:
        Residual probability matrix, defined as ``Q - Q_plus``.
    CM:
        Standard hard confusion matrix, ``T.T @ P``.
    CM_star:
        Probability-mass confusion matrix, ``T.T @ Q``.
    V:
        Decisive probability-mass matrix, ``T.T @ Q_plus``.
    U:
        Residual-dispersion matrix, ``T.T @ Q_minus``.
    lambda_v:
        Mean top-1 probability, equivalently ``sum(V) / n``.
    lambda_u:
        Mean residual probability mass, equivalently ``sum(U) / n``.
    accuracy:
        Hard-label accuracy, ``trace(CM) / n``.
    accuracy_star:
        Probability-mass accuracy, ``trace(CM_star) / n``.
    """

    classes: NDArray[Any]
    y_index: IntVector
    predicted_index: IntVector
    T: FloatMatrix
    P: FloatMatrix
    Q: FloatMatrix
    Q_plus: FloatMatrix
    Q_minus: FloatMatrix
    CM: FloatMatrix
    CM_star: FloatMatrix
    V: FloatMatrix
    U: FloatMatrix
    lambda_v: float
    lambda_u: float
    accuracy: float
    accuracy_star: float

    def scalar_summary(self) -> dict[str, float]:
        """Return the four scalar summaries as a plain dictionary."""
        return {
            "Acc": self.accuracy,
            "Acc_star": self.accuracy_star,
            "lambda_v": self.lambda_v,
            "lambda_u": self.lambda_u,
        }

    def matrix_summary(self) -> dict[str, FloatMatrix]:
        """Return all framework matrices in a name-to-array dictionary."""
        return {
            "T": self.T,
            "P": self.P,
            "Q": self.Q,
            "Q_plus": self.Q_plus,
            "Q_minus": self.Q_minus,
            "CM": self.CM,
            "CM_star": self.CM_star,
            "V": self.V,
            "U": self.U,
        }


def validate_probability_matrix(
    probabilities: ArrayLike,
    *,
    atol: float = 1e-10,
) -> FloatMatrix:
    """Validate and return a two-dimensional class-probability matrix.

    Parameters
    ----------
    probabilities:
        Array of shape ``(n_samples, n_classes)``. Every entry must lie in
        ``[0, 1]`` and every row must sum to one.
    atol:
        Absolute tolerance used for the row-sum check.

    Returns
    -------
    numpy.ndarray
        A defensive floating-point copy of the validated matrix.

    Raises
    ------
    ValueError
        If the matrix has an invalid shape, contains non-finite values, has
        entries outside ``[0, 1]``, or contains rows that do not sum to one.
    """
    Q = np.asarray(probabilities, dtype=float)
    if Q.ndim != 2:
        raise ValueError("The probability matrix Q must be two-dimensional.")
    if Q.shape[0] == 0:
        raise ValueError("Q must contain at least one instance.")
    if Q.shape[1] < 2:
        raise ValueError("Q must contain at least two class columns.")
    if not np.isfinite(Q).all():
        raise ValueError("Q contains NaN or infinite values.")
    if np.any(Q < -atol) or np.any(Q > 1.0 + atol):
        raise ValueError("Every probability in Q must lie in [0, 1].")

    Q = np.clip(Q, 0.0, 1.0)
    if not np.allclose(Q.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("Every row of Q must sum to one.")
    return Q.copy()


def encode_true_labels(
    y_true: Sequence[Any] | ArrayLike,
    n_classes: int,
    classes: Iterable[Any] | None = None,
) -> tuple[IntVector, NDArray[Any]]:
    """Encode true labels according to the probability-column order.

    Parameters
    ----------
    y_true:
        Sequence containing one true label per probability row.
    n_classes:
        Number of columns in ``Q``.
    classes:
        Labels corresponding, in order, to the columns of ``Q``. This argument
        is required for non-integer labels. When omitted, ``y_true`` must
        already contain integer indices in ``0, ..., n_classes - 1``.

    Returns
    -------
    y_index, classes
        Integer-encoded labels and the validated class-label array.
    """
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if classes is None:
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                "classes must be provided when y_true does not contain integer "
                "indices matching the columns of Q."
            )
        y_index = y.astype(int, copy=True)
        class_array = np.arange(n_classes)
    else:
        class_array = np.asarray(list(classes), dtype=object)
        if class_array.ndim != 1 or class_array.size != n_classes:
            raise ValueError(
                "classes must contain exactly one label for each column of Q."
            )
        if len(set(class_array.tolist())) != n_classes:
            raise ValueError("Class labels must be unique.")
        lookup = {label: index for index, label in enumerate(class_array.tolist())}
        try:
            y_index = np.asarray([lookup[label] for label in y.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown true label: {exc.args[0]!r}.") from exc

    if np.any(y_index < 0) or np.any(y_index >= n_classes):
        raise ValueError("At least one encoded true label is outside the valid range.")
    return y_index, class_array


def one_hot(indices: ArrayLike, n_classes: int) -> FloatMatrix:
    """Create a one-hot matrix from integer class indices."""
    encoded = np.asarray(indices, dtype=int)
    if encoded.ndim != 1:
        raise ValueError("Class indices must be one-dimensional.")
    if np.any(encoded < 0) or np.any(encoded >= n_classes):
        raise ValueError("At least one class index is outside the valid range.")

    matrix = np.zeros((encoded.size, n_classes), dtype=float)
    matrix[np.arange(encoded.size), encoded] = 1.0
    return matrix


def build_prediction_matrix(Q: ArrayLike) -> tuple[FloatMatrix, IntVector]:
    """Build the hard prediction matrix ``P`` from probability matrix ``Q``.

    NumPy's ``argmax`` is used as a deterministic tie rule; therefore, the first
    maximum in class-column order is selected when two or more probabilities
    are equal.
    """
    validated = validate_probability_matrix(Q)
    predicted_index = np.argmax(validated, axis=1).astype(int)
    return one_hot(predicted_index, validated.shape[1]), predicted_index


def decompose_probability_matrix(
    Q: ArrayLike,
) -> tuple[FloatMatrix, FloatMatrix, IntVector]:
    """Decompose ``Q`` into selected top-1 and residual probability mass.

    Returns
    -------
    Q_plus, Q_minus, top1_index
        ``Q_plus`` retains one tie-resolved maximum per row, ``Q_minus`` is the
        complementary residual matrix, and ``top1_index`` records the selected
        class column.
    """
    validated = validate_probability_matrix(Q)
    top1_index = np.argmax(validated, axis=1).astype(int)
    Q_plus = np.zeros_like(validated)
    rows = np.arange(validated.shape[0])
    Q_plus[rows, top1_index] = validated[rows, top1_index]
    Q_minus = validated - Q_plus
    return Q_plus, Q_minus, top1_index


def build_confusion_matrix(T: ArrayLike, P: ArrayLike) -> FloatMatrix:
    """Compute the standard confusion matrix ``CM = T.T @ P``."""
    T_array = np.asarray(T, dtype=float)
    P_array = np.asarray(P, dtype=float)
    if T_array.ndim != 2 or P_array.ndim != 2:
        raise ValueError("T and P must be two-dimensional matrices.")
    if T_array.shape != P_array.shape:
        raise ValueError("T and P must have identical shapes.")
    return T_array.T @ P_array


def build_probabilistic_confusion_matrix(
    T: ArrayLike,
    Q: ArrayLike,
) -> FloatMatrix:
    """Compute the probability-mass confusion matrix ``CM_star = T.T @ Q``."""
    T_array = np.asarray(T, dtype=float)
    Q_array = validate_probability_matrix(Q)
    if T_array.ndim != 2 or T_array.shape != Q_array.shape:
        raise ValueError("T and Q must be two-dimensional matrices of equal shape.")
    return T_array.T @ Q_array


def decompose_probabilistic_confusion_matrix(
    T: ArrayLike,
    Q_plus: ArrayLike,
    Q_minus: ArrayLike,
) -> tuple[FloatMatrix, FloatMatrix]:
    """Compute the decisive and residual matrices ``V`` and ``U``."""
    T_array = np.asarray(T, dtype=float)
    plus = np.asarray(Q_plus, dtype=float)
    minus = np.asarray(Q_minus, dtype=float)
    if T_array.ndim != 2 or plus.ndim != 2 or minus.ndim != 2:
        raise ValueError("T, Q_plus, and Q_minus must be two-dimensional.")
    if T_array.shape != plus.shape or plus.shape != minus.shape:
        raise ValueError("T, Q_plus, and Q_minus must have identical shapes.")
    return T_array.T @ plus, T_array.T @ minus


def compute_weights(V: ArrayLike, U: ArrayLike) -> tuple[float, float]:
    """Compute ``lambda_v`` and ``lambda_u`` from matrices ``V`` and ``U``."""
    V_array = np.asarray(V, dtype=float)
    U_array = np.asarray(U, dtype=float)
    if V_array.shape != U_array.shape or V_array.ndim != 2:
        raise ValueError("V and U must be two-dimensional matrices of equal shape.")
    total = float((V_array + U_array).sum())
    if total <= 0.0:
        raise ValueError("The total probability mass must be positive.")
    lambda_v = float(V_array.sum() / total)
    lambda_u = float(U_array.sum() / total)
    return lambda_v, lambda_u


def compute_accuracy(CM: ArrayLike) -> float:
    """Compute hard-label accuracy as ``trace(CM) / sum(CM)``."""
    matrix = np.asarray(CM, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("CM must be a square matrix.")
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("CM must contain positive total mass.")
    return float(np.trace(matrix) / total)


def compute_probabilistic_accuracy(CM_star: ArrayLike) -> float:
    """Compute ``Acc_star`` as ``trace(CM_star) / sum(CM_star)``."""
    matrix = np.asarray(CM_star, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("CM_star must be a square matrix.")
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("CM_star must contain positive total probability mass.")
    return float(np.trace(matrix) / total)


def evaluate_probabilities(
    y_true: Sequence[Any] | ArrayLike,
    probabilities: ArrayLike,
    *,
    classes: Iterable[Any] | None = None,
    atol: float = 1e-10,
) -> ProbabilityMassEvaluation:
    """Compute all matrices and scalar measures from instance probabilities.

    Parameters
    ----------
    y_true:
        True labels, one per row of ``probabilities``.
    probabilities:
        Matrix ``Q`` with shape ``(n_samples, n_classes)``.
    classes:
        Labels corresponding to the columns of ``Q``. Provide this argument for
        string or otherwise non-index labels.
    atol:
        Absolute tolerance used in probability and identity checks.

    Returns
    -------
    ProbabilityMassEvaluation
        Immutable object containing ``T``, ``P``, ``Q``, ``Q_plus``,
        ``Q_minus``, ``CM``, ``CM_star``, ``V``, ``U``, ``lambda_v``,
        ``lambda_u``, ``Acc``, and ``Acc_star``.
    """
    Q = validate_probability_matrix(probabilities, atol=atol)
    y_index, class_array = encode_true_labels(y_true, Q.shape[1], classes)
    if y_index.size != Q.shape[0]:
        raise ValueError("y_true and Q must contain the same number of instances.")

    T = one_hot(y_index, Q.shape[1])
    P, predicted_index = build_prediction_matrix(Q)
    Q_plus, Q_minus, decomposition_index = decompose_probability_matrix(Q)
    if not np.array_equal(predicted_index, decomposition_index):
        raise AssertionError("Hard predictions and Q decomposition use different ties.")

    CM = build_confusion_matrix(T, P)
    CM_star = build_probabilistic_confusion_matrix(T, Q)
    V, U = decompose_probabilistic_confusion_matrix(T, Q_plus, Q_minus)

    if not np.allclose(Q, Q_plus + Q_minus, atol=atol, rtol=0.0):
        raise AssertionError("Q != Q_plus + Q_minus within the requested tolerance.")
    cm_tolerance = max(atol, 128.0 * np.finfo(float).eps * Q.shape[0])
    if not np.allclose(CM_star, V + U, atol=cm_tolerance, rtol=0.0):
        raise AssertionError("CM_star != V + U within numerical tolerance.")

    lambda_v, lambda_u = compute_weights(V, U)
    accuracy = compute_accuracy(CM)
    accuracy_star = compute_probabilistic_accuracy(CM_star)

    direct_lambda_v = float(np.mean(np.max(Q, axis=1)))
    direct_accuracy_star = float(np.mean(Q[np.arange(Q.shape[0]), y_index]))
    if not np.isclose(lambda_v, direct_lambda_v, atol=atol, rtol=0.0):
        raise AssertionError("Equivalent definitions of lambda_v disagree.")
    if not np.isclose(lambda_v + lambda_u, 1.0, atol=atol, rtol=0.0):
        raise AssertionError("lambda_v + lambda_u != 1 within tolerance.")
    if not np.isclose(accuracy_star, direct_accuracy_star, atol=atol, rtol=0.0):
        raise AssertionError("Equivalent definitions of Acc_star disagree.")

    return ProbabilityMassEvaluation(
        classes=class_array,
        y_index=y_index,
        predicted_index=predicted_index,
        T=T,
        P=P,
        Q=Q,
        Q_plus=Q_plus,
        Q_minus=Q_minus,
        CM=CM,
        CM_star=CM_star,
        V=V,
        U=U,
        lambda_v=lambda_v,
        lambda_u=lambda_u,
        accuracy=accuracy,
        accuracy_star=accuracy_star,
    )


__all__ = [
    "ProbabilityMassEvaluation",
    "build_confusion_matrix",
    "build_prediction_matrix",
    "build_probabilistic_confusion_matrix",
    "compute_accuracy",
    "compute_probabilistic_accuracy",
    "compute_weights",
    "decompose_probability_matrix",
    "decompose_probabilistic_confusion_matrix",
    "encode_true_labels",
    "evaluate_probabilities",
    "one_hot",
    "validate_probability_matrix",
]
