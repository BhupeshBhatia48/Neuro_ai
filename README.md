# 🧠 Neuro AI — Autonomous ML Engineer

> Give it an idea. It finds a dataset, preprocesses it, trains the best model, evaluates it, and exports everything — fully automated.

---

## Quick Start

```bash
# 1. Install core dependencies
pip install -r requirements/requirements-core.txt

# 2. Set your API key (optional — works fully offline too)
cp .env.example .env
# edit .env and add OPENROUTER_API_KEY

# 3. Run
python main.py --idea "Predict customer churn for a telecom company"

# Offline mode (no LLM key needed)
python main.py --idea "Classify iris flowers" --mode manual

# Direct URL — skip search entirely
python main.py --url https://raw.githubusercontent.com/datasets/iris/main/data/iris.csv --target species
```

---

## 📦 Requirements — Modular Install

```bash
# Minimal (tabular ML only)
pip install -r requirements/requirements-core.txt

# + AutoML (FLAML)
pip install -r requirements/requirements-automl.txt

# + Deep Learning (PyTorch, Transformers) — EXPERIMENTAL
pip install -r requirements/requirements-deep.txt

# + UI/API (Streamlit, FastAPI)
pip install -r requirements/requirements-ui.txt
```

---

## 🔀 Model Routing Strategy

```
User Idea / Dataset URL
        │
        ▼
 ┌─────────────────────┐
 │   Idea Parser       │  LLM (OpenRouter) OR rule-based fallback
 │   (Issue #1, #9)    │  USE_LLM=false → pure offline keyword rules
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Multi-Source Search│  DuckDuckGo + OpenML + HuggingFace + Kaggle
 │  (Issue #3, #13)    │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Dataset Quality    │  Score 0–100 (missing %, imbalance, variance)
 │  (Issue #10)        │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Preprocessing      │  Split FIRST → fit on train → transform test
 │  (Issue #4)         │  Zero data leakage guaranteed
 └────────┬────────────┘
          │
     ─────┴──────────────────────────────────────────
     │                     │                        │
     ▼                     ▼                        ▼
┌──────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  --mode  │     │    --mode auto   │     │  DL model requested │
│  manual  │     │   (default)      │     │  (LSTM/GRU/MLP/etc) │
└────┬─────┘     └────────┬─────────┘     └──────────┬──────────┘
     │                    │                          │
     ▼                    ▼                          ▼
┌──────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Model   │     │  FLAML AutoML    │     │  DeepLearningTrainer│
│  Trainer │     │  (Issue #6, #8)  │     │  ⚠ EXPERIMENTAL     │
│  sklearn │     │  Best of 10+     │     │  GPU-aware, batched  │
│  +boost  │     │  estimators      │     │  (Issue #2, #14)    │
└────┬─────┘     └────────┬─────────┘     └──────────┬──────────┘
     │                    │                          │
     └────────────────────┴──────────────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Evaluator       │  Leaderboard, plots, metrics
               │  + Optuna Tuning │  (optional --tune flag)
               └──────────┬───────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Export          │  model.pkl + preprocessing.pkl
               │  + Report        │  training_report.txt + charts
               └──────────────────┘
```

### Rule-Based Recommendation Flow
When `USE_LLM=false` or no API key is set, idea parsing uses keyword matching:
- `forecast / trend / time series` → `time_series`
- `image / photo / detect` → `computer_vision`
- `text / sentiment / NLP` → `nlp`
- `cluster / segment` → `clustering`
- `regression / price / salary` → `regression`
- default → `classification`

### AutoML Decision Flow (Issue #6)
| AUTOML_MODE | What runs | Best for |
|---|---|---|
| `auto` (default) | FLAML — searches 10+ estimators automatically | Unknown datasets, want best accuracy |
| `manual` | ModelTrainer — trains your specified model list | Known good model, faster iteration |

### Manual Override Flow
```bash
# Force specific model
python main.py --idea "..." --model XGBoost --mode manual

# Force AutoML
python main.py --idea "..." --mode auto

# Deep learning (EXPERIMENTAL)
python main.py --idea "..." --model LSTM --mode manual
```

---

## ⚠ Deep Learning — Experimental (Issue #2)

Deep learning support is marked **EXPERIMENTAL**. It works but is not recommended for production pipelines.

| Model | Task | Status |
|---|---|---|
| LSTM | time_series, classification | ✅ Experimental |
| GRU | time_series, classification | ✅ Experimental |
| MLP | classification, regression | ✅ Experimental |
| BERT / DistilBERT | nlp | 🔜 Planned |
| ResNet / EfficientNet | computer_vision | 🔜 Planned |

Install DL deps: `pip install -r requirements/requirements-deep.txt`

---

## 🔑 Environment Variables (.env)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter key — get at openrouter.ai/keys |
| `OPENROUTER_MODEL` | `openrouter/auto` | Any model slug from openrouter.ai/models |
| `USE_LLM` | `true` | Set `false` for fully offline mode |
| `AUTOML_MODE` | `auto` | `auto` (FLAML) or `manual` |
| `FLAML_TIME_BUDGET` | `120` | Seconds for FLAML AutoML |
| `MAX_DATASET_ROWS` | `300000` | Sample larger datasets to this size |
| `DEFAULT_RANDOM_STATE` | `42` | Global seed for reproducibility |
| `KAGGLE_USERNAME` | — | Kaggle credentials (optional) |
| `KAGGLE_KEY` | — | Kaggle API key (optional) |

---

## 📁 Project Structure

```
neuro-ai/
├── config/              # Settings + constants
├── llm_agent/           # OpenRouter client, idea parser, keyword extractor
├── search_engine/       # DuckDuckGo + OpenML + HuggingFace + Kaggle search
├── dataset_engine/      # Download, load, validate datasets
├── data_analysis/       # Profile, feature detection, target detection
├── preprocessing/       # Leakage-safe pipeline builder
├── model_engine/        # Candidate models, trainer, evaluator, tuner
├── automl/              # FLAML AutoML controller
├── deep_learning/       # ⚠ EXPERIMENTAL: LSTM/GRU/MLP trainer
├── export_engine/       # Pickle exporter, report generator
├── requirements/        # Modular requirements files
│   ├── requirements-core.txt
│   ├── requirements-automl.txt
│   ├── requirements-deep.txt
│   └── requirements-ui.txt
├── main.py              # Pipeline orchestrator
└── .env.example         # Environment variable template
```
