# Calculating Beta in Python

A hands-on lesson for early to mid-career finance professionals on calculating a stock's **beta** from historical price data using `pandas` and `statsmodels`.

We regress daily log returns of 3 household-name stocks over a full economic cycle (July 2019 – June 2026), then interpret the OLS output: what beta means, how the regression slope maps to systematic risk, why the betas of 3 companies differ, and how beta feeds the Capital Asset Pricing Model (CAPM) cost of equity.

## What's in this repo

| File | Description |
|------|-------------|
| `calculating_beta_student.ipynb` | **Start here.** Follow-along version with `TODO` cells to complete during the session. |
| `calculating_beta_instructor.ipynb` | Completed version with all code and outputs — the answer key. |
| `data/` | Cached price CSVs so the notebooks run even without internet access. |
| `pyproject.toml` / `uv.lock` | Project dependencies, managed with [uv](https://docs.astral.sh/uv/). |
| `build_notebooks.py` | Generates both notebooks from a single source (instructor use only). |

## Setup (before the session)

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) to manage both the Python installation and the project libraries — no separate Python download needed.

First check whether you already have it:

```bash
uv --version
```

If that prints a version number (e.g. `uv 0.9.27`), skip ahead to step 2. If you get "command not found", install it:

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal afterwards, then re-run `uv --version` to confirm the install worked.

### 2. Clone the repo and install everything

```bash
git clone https://github.com/dgodevais/wallstreetprep-calculating-beta-python-lesson.git
cd wallstreetprep-calculating-beta-python-lesson

uv sync
```

`uv sync` reads `pyproject.toml`, downloads a compatible Python 3.12+ interpreter if you don't have one, creates a virtual environment in `.venv/`, and installs the exact library versions pinned in `uv.lock`.

### 3. Launch the notebook

```bash
uv run jupyter lab calculating_beta_student.ipynb
```

Run the first two code cells (imports and data download) to confirm everything works. If the download fails, the notebook automatically falls back to the CSVs in `data/`.

## What we cover (~25 minutes)

1. What beta is — systematic risk and the CAPM
2. Pulling adjusted OHLC data with `yfinance` for all three stocks
3. Computing log returns (and why log, not simple)
4. Visualizing each stock vs. the market — three clouds, three slopes
5. Running the OLS regressions with `statsmodels`
6. Interpreting the output — why Carnival ≈ 2, Disney ≈ 1, and Coca-Cola ≈ 0.5
7. Sanity-checking with the cov/var formula + rolling betas through COVID
