# Flood Risk Capital Allocator

An interactive decision-support simulator that translates K-Nearest Neighbors (`k = 17`) flood-probability predictions into an optimal capital-allocation threshold.

Standard machine learning classifiers default to a 0.50 probability cutoff, which treats false positives and false negatives as equally costly. In infrastructure planning, deploying flood-prevention capital in a zone that does not flood wastes the unit intervention budget, whereas failing to protect a zone that floods exposes the municipality or insurer to much larger property damage. This application evaluates 19 candidate probability cutoffs (`0.05` to `0.95`) against user-defined economic parameters to identify the threshold that maximizes expected **Net Benefit** and **Return on Investment (ROI)**.

## Economic formulation

For any candidate probability threshold $p \in [0.05, 0.95]$, let $N_{\text{flagged}}(p)$ be the number of zones with predicted flood probability $\hat{y}_i \ge p$, and let $TP(p)$ be the number of true flood events among those flagged zones:

$$\text{Intervention Cost}(p) = N_{\text{flagged}}(p) \times C_{\text{area}}$$

$$\text{Prevented Damage}(p) = TP(p) \times D_{\text{flood}} \times \eta$$

$$\text{Net Benefit}(p) = \text{Prevented Damage}(p) - \text{Intervention Cost}(p)$$

where $C_{\text{area}}$ is the capital expenditure per intervention zone (\$0.5M to \$5.0M), $D_{\text{flood}}$ is the expected economic loss per unmitigated flood (\$2.0M to \$20.0M), and $\eta \in [0.40, 1.00]$ is the intervention effectiveness rate.

## Model and feature set

The serialized `KNeighborsClassifier` artifact (`models/knn_model.pkl`) is trained on 800 observations and evaluated on a held-out partition of 200 zones across 16 hydrological, topographic, infrastructure, and institutional predictors:

| Category | Features |
|---|---|
| Topographic & Hydrological | `latitude`, `longitude`, `elevation_m`, `elev_local_delta_knn`, `drainage_density_km_per_km2`, `return_period_years`, `log_historical_rainfall_intensity_mm_hr`, `soil_group_B` |
| Infrastructure & Land Use | `log_storm_drain_proximity_m`, `storm_drain_type_Combined`, `land_use_Urban` |
| Socioeconomic & Governance | `gdp_per_capita`, `population_density`, `rule_of_law`, `control_of_corruption`, `region_Europe` |

## Architecture

```text
.
├── .devcontainer/devcontainer.json  # GitHub Codespaces environment
├── .github/workflows/ci.yml         # Automated Ruff linting and Pytest suite
├── .streamlit/config.toml           # Theme configuration
├── assets/
│   └── styles.css                   # Application stylesheet
├── flood_risk/
│   ├── __init__.py
│   ├── charts.py                    # Matplotlib figure builders for cost-benefit and PR curves
│   ├── model_loader.py              # Artifact resolution and k=17 KNN loader
│   └── simulation.py                # Pure threshold-optimization and ROI calculation engine
├── models/
│   └── knn_model.pkl                # Trained KNN classifier and 200-zone evaluation split
├── tests/
│   └── test_simulation.py           # Unit tests for financial math, edge cases, and artifact presence
├── app.py                           # Streamlit entry point
├── pyproject.toml                   # Package metadata, Ruff, and Pytest settings
└── requirements.txt                 # Pinned runtime dependencies
```

## Local setup

Create a Python 3.11+ virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

## Running the test suite

```bash
python3 -m unittest discover -s tests -v
```

Or with `pytest` and `ruff`:

```bash
ruff check .
pytest -v
```

## Author

Built by **Zacharie Houle, CPA auditeur**. Licensed under the [MIT License](LICENSE).
