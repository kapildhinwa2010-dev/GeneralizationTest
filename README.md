# GeneralizationTest

GeneralizationTest is a small NumPy utility for comparing the singular-value structure of two-dimensional layer matrices. Its seen-versus-unseen comparison is a **spectral proxy for investigation**, not a measurement of model generalization or overfitting.

## Status

The project is version `0.1.0` and contains a single Python module with unit tests. It implements numerical matrix-analysis helpers, but it does not include a neural-network integration, dataset loader, training or evaluation pipeline, or empirical research results. Treat its output as exploratory diagnostics rather than evidence that a model generalizes or overfits.

## Implemented functionality

- Compute descending singular values for a two-dimensional matrix.
- Estimate effective rank at a selected cumulative singular-value threshold (default `0.99`).
- Report matrix shape, Frobenius norm, spectral norm, condition number, and effective rank for one matrix.
- Analyze a mapping of layer names to matrices and rank layers by effective rank.
- Compare named seen and unseen matrices by effective-rank differences at thresholds `0.5`, `0.9`, and `0.99`, plus a norm ratio; identify the layer with the largest summed rank-gap score.

Inputs are converted to `float64`. The implementation expects non-empty mappings where required and two-dimensional matrices. `generalization_report` checks that the seen and unseen mappings have the same layer names; it does not validate that corresponding matrices have identical shapes.

## Requirements and installation

Python 3.9 or newer and NumPy 1.24 or newer are declared in `pyproject.toml`. Install the project from its root directory with:

```bash
python -m pip install .
```

To install the dependencies used by the test suite from the checkout instead:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from generalizationtest import analyze_model, find_bottlenecks, generalization_report

layers = {
    "projection": np.diag([4.0, 2.0, 1.0, 0.0]),
    "output": np.eye(4),
}
reports = analyze_model(layers)
print(reports["projection"])
print(find_bottlenecks(reports, top_k=1))

comparison = generalization_report(
    {"projection": np.diag([4.0, 2.0, 1.0, 0.0])},
    {"projection": np.diag([4.0, 0.0, 0.0, 0.0])},
)
print(comparison["most_unstable"])
print(comparison["layers"]["projection"])
```

The caller must provide matrices that meaningfully represent the intended seen and unseen quantities; this module does not extract weights or activations from a model or construct those matrices from data.

## Tests and CI

On the checked-out commit `5641908acdbfc64c2f11961ed7ee41d87034bc55`, the test suite was run locally with:

```bash
python -m pytest -q
```

Result: **17 passed**. This is a local result for that commit, not a claim about other commits or remote CI runs.

The repository's GitHub Actions CI workflow is configured to run pytest, compile the module and tests, and perform an import/report smoke check on Python 3.10, 3.11, and 3.12. The CodeQL workflow is configured for Python analysis; its remote execution status has not been verified here.

## Limitations

- The code measures matrix spectra and rank-structure changes only. It does not compute train-versus-held-out losses, an empirical generalization gap, accuracy, or proof of overfitting.
- It does not load, inspect, or execute neural-network models; callers must provide the matrices themselves.
- It includes no benchmarks or reported experimental results.
- The `spectral_norm_ratio` result field in `spectrum_gap` is calculated using NumPy's default matrix norm (`np.linalg.norm` without an order), which is the Frobenius norm for a matrix, not the spectral norm.
- `effective_rank` uses cumulative singular-value magnitudes, not squared singular values, for its threshold calculation.
- Corresponding seen/unseen matrix shapes are not explicitly checked by `generalization_report`.

## License

MIT License. See [LICENSE](LICENSE) for the copyright notice and full terms.
