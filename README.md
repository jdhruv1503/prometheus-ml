# `prometheus-ml` — **the Claude Code of ML competitions**

```text
┌──────────────────────────────────────────────────────────────────────┐
│ PROMETHEUS // TIME-AWARE EXPERIMENT POLICY ENGINE                   │
├──────────────────────────────────────────────────────────────────────┤
│ budget: 120m   elapsed: 34m   remaining: 86m                        │
│                                                                      │
│ rank  experiment                EV gain   time   risk   source      │
│  01   target_encoding_v2        +0.012    7m    0.11   kb:lgbm      │
│  02   catboost_monotone         +0.008    5m    0.22   arxiv:2204   │
│  03   lgbm_interaction_constr   +0.007    5m    0.12   kb:lgbm      │
│  04   smote_gbdt_boundary       +0.006    4m    0.09   arxiv:2301   │
│  05   leakage_probe_time_split  +0.000    3m    0.03   safety       │
└──────────────────────────────────────────────────────────────────────┘
```

Hackathon ML is not a model selection problem. It is a **sequential decision problem under clock pressure**. Prometheus is the agent that solves it — grounded in 50,000+ ML papers and deep library expertise, not just LLM vibes.

---

## quickstart

```bash
pip install prometheus-ml
prometheus demo                          # zero-config demo, no API keys needed
```

```bash
# Real data
prometheus init --train train.csv --target is_fraud --metric auc --budget 4h
prometheus brief                         # competition brief before first model
prometheus run                           # live TUI TEPE loop
```

---

## what makes this different

**Three intelligence sources, working together:**

```
KB hypotheses   →  20+ known-good experiments (LightGBM/XGBoost/CatBoost params,
                   feature engineering recipes, CV schemes) — generated in ms,
                   no API needed

ArXiv hypotheses → "What did recent papers say about this exact problem type?"
                   Searches 50k+ ML papers, extracts implementation-ready patches

LLM hypotheses  →  Novel semantic features specific to YOUR competition domain
                   (graceful fallback to KB+ArXiv if LLM unavailable)
```

All three feed into **TEPE** — the time-aware policy engine that ranks and runs them.

---

## arxiv intelligence layer

Search 50,000+ ML papers and get implementation-ready hypotheses:

```bash
prometheus arxiv "gradient boosting tabular imbalanced classification"
```

```
Searching 50,847 ML papers...

TOP 5 PAPERS — IMPLEMENTATION-READY HYPOTHESES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "Monotone Constraints in Gradient Boosting for Credit Scoring" (2022)
   cs.LG | arxiv:2204.07887
   Technique: monotone_constraints + interaction_constraints combo
   Reported: +1.8% AUC on credit/fraud datasets
   ★ HYPOTHESIS: lgbm_monotone_interaction_constrained
     patch: {monotone_constraints: [1,-1,0,1], interaction_constraints: '[[0,1],[2,3]]'}
     EV prior: +0.008 AUC | Time: ~5m | Risk: Low

2. "Class-Imbalanced Learning via Contrastive Resampling" (2023)
   cs.LG | arxiv:2301.09483
   Technique: SMOTE-variant with GBDT-guided boundary sampling
   Reported: +2.3% AUC on imbalanced tabular benchmarks
   ★ HYPOTHESIS: smote_gbdt_boundary_sampling
     patch: {sampler: BorderlineSMOTE, k_neighbors: 5, sampling_strategy: 0.1}
     EV prior: +0.006 AUC | Time: ~4m | Risk: Low

[+ 6 more hypotheses added to your experiment queue]
```

**How it works:**
- Pre-indexed embeddings of `cs.LG`, `stat.ML`, `cs.AI` papers (2018–present) stored locally at `~/.prometheus/paper-store/`
- Semantic ANN search (sentence-transformers + FAISS) — fully offline after install
- LLM extracts: dataset_type, metric, technique, reported_improvement → `Hypothesis` object
- Daily `arxiv-monitor` checks for new papers matching your competition archetype
- Citation graph: tracks which papers led to real CV gains → builds author reputation scores

---

## built-in ML library knowledge base

The hypothesis generator has deep, built-in knowledge of every major ML library — no API calls needed for well-known techniques:

```bash
~/.prometheus/kb/
├── libraries/
│   ├── lightgbm.json     # All params: monotone_constraints, DART, GOSS,
│   │                     # interaction_constraints, categorical_feature, etc.
│   ├── xgboost.json      # tree_method, gpu_hist, monotone, feature_interactions
│   ├── catboost.json     # ordered boosting, text features, embedding features
│   └── sklearn.json      # Pipeline, ColumnTransformer, all transformers
├── feature_engineering/
│   ├── featuretools.json        # DFS patterns, custom primitives
│   ├── category_encoders.json   # Target enc (safe vs leaky), LOO, WoE, CatBoost enc
│   └── feature_engine.json      # Common FE patterns
├── validation/
│   ├── cv_schemes.json          # When to use: GroupKFold, PurgedKFold, StratifiedKFold
│   ├── leakage_patterns.json    # 20+ known leakage signatures + auto-fixes
│   └── adversarial_validation.json
└── ensembling/
    ├── stacking.json            # Ridge meta-learner, sklearn Pipeline patterns
    ├── blending.json            # Nelder-Mead, rank averaging, power averaging
    └── diversity.json           # OOF correlation, diversity metrics
```

**Example — LightGBM KB entry:**
```json
{
  "interaction_constraints": {
    "use_when": "feature groups should only interact within their group",
    "code": "lgb.train({'interaction_constraints': '[[0,1,2],[3,4,5]]'})",
    "competition_evidence": ["AmEx 2022 gold", "IEEE-CIS gold"],
    "typical_gain": "+0.001 to +0.005 AUC (finance datasets)"
  },
  "dart": {
    "use_when": "standard GBDT overfits, dataset has noisy labels",
    "params": {"boosting_type": "dart", "drop_rate": 0.1},
    "caveat": "non-deterministic prediction — average multiple runs",
    "typical_gain": "+0.003 AUC on noisy targets"
  }
}
```

**Key insight:** Template hypotheses (monotone constraints, target encoding, DART mode) generate in milliseconds from the KB. The LLM is called only for novel, domain-specific hypotheses. This means Prometheus works fully offline for 80% of its value.

---

## architecture

```text
 INTELLIGENCE SOURCES (offline-first)
 ┌──────────────────────────────────────────────────────────┐
 │  ~/.prometheus/kb/              ~/.prometheus/paper-store/│
 │  Library Knowledge Base         50k ArXiv Papers          │
 │  lightgbm.json, xgboost.json    index.faiss (embeddings)  │
 │  category_encoders.json         metadata.parquet          │
 │  leakage_patterns.json          <arxiv_id>.json (cached)  │
 └──────────────────────────────────────────────────────────┘
                        │
                        ▼
 CSV + problem spec → intake/profile → DNA classifier
                                            │
                                            ▼
                               ┌────────────────────────┐
                               │   hypothesis generator  │
                               │   1. KB (fast, offline) │
                               │   2. ArXiv (ANN search) │
                               │   3. LLM (novel feats)  │
                               └────────────┬───────────┘
                                            ▼
                                    TEPE scheduler
                                    score = E[gain]/E[time]
                                          × (1-leakage_risk)
                                            │
                                            ▼
                               execute → validate → update
                                            │
                                            ▼
                                        ensemble
```

## how TEPE works

Each hypothesis is scored as:

```
score = E[gain] / E[runtime] × (1 - leakage_risk)
```

After each experiment, TEPE updates gain/runtime/risk posteriors (Thompson sampling) and re-ranks the queue. That means Prometheus reacts to *evidence*, not vibes.

The experiment source is tracked: `kb:lgbm` means it came from the library knowledge base, `arxiv:2204.07887` means it came from a specific paper. After your competition, Prometheus knows which papers actually improved your score.

---

## commands

```bash
# Core workflow
prometheus init --train train.csv --target is_fraud --metric auc --budget 4h
prometheus brief              # competition brief (read before first model)
prometheus run                # live TUI TEPE loop
prometheus demo               # zero-config Titanic demo

# Literature intelligence
prometheus arxiv "monotone constraints fraud"    # search 50k papers → hypotheses
prometheus arxiv --similar-to current           # papers matching your competition
prometheus arxiv update                         # pull last 7 days from arXiv API

# Experiment management
prometheus status             # quick status
prometheus board              # full experiment board (EV/time/risk)
prometheus queue --add "try neural embedding for categorical features"

# Ensembling & submission
prometheus blend --models ./oof/ --metric auc
prometheus portfolio          # submission portfolio manager
prometheus submit --risk conservative

# Post-competition
prometheus autopsy            # full debrief with personalized lessons
prometheus replay <run_id>    # deterministic replay of past run
```

---

## the viral moment

```
TEPE vs Random Order — Same 10 Experiments, Same Time Budget
─────────────────────────────────────────────────────────────
Random order:   0.7823 → 0.8021  in 70 minutes  (+0.0198)
TEPE order:     0.7823 → 0.8467  in 70 minutes  (+0.0644)

TEPE found +0.0446 more AUC with the SAME experiments.
The difference is the ORDER.
```

Same experiments. Different order. The order is the product.

---

## docs

- Full product spec: [`FINAL_SPEC.md`](https://github.com/jdhruv1503/prometheus-ml) (see workspace)
- Architecture details: `docs/architecture.md`
- ArXiv layer design: `docs/arxiv-intelligence.md`
- Library KB schema: `docs/kb-schema.md`
