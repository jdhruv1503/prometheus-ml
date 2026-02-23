# `prometheus-ml` — **stop guessing your next experiment**

```text
┌──────────────────────────────────────────────────────────────────────┐
│ PROMETHEUS // TIME-AWARE EXPERIMENT POLICY ENGINE                   │
├──────────────────────────────────────────────────────────────────────┤
│ budget: 120m   elapsed: 34m   remaining: 86m                        │
│                                                                      │
│ rank  experiment                EV gain   time   risk   score        │
│  01   target_encoding_v2        +0.012    7m    0.11   0.00153      │
│  02   catboost_monotone         +0.008    5m    0.22   0.00125      │
│  03   leakage_probe_time_split  +0.000    3m    0.03   0.00031      │
└──────────────────────────────────────────────────────────────────────┘
```

Hackathon ML is not a model selection problem. It is a **sequential decision problem under clock pressure**.

## quickstart

```bash
git clone https://github.com/jdhruv1503/prometheus-ml
cd prometheus-ml
python -m venv .venv && source .venv/bin/activate
pip install -e .
prometheus init --dataset tests/fixtures/sample.csv --target survived --metric accuracy --budget 10
```

## architecture

```text
 CSV + problem spec
        │
        ▼
 ┌───────────────┐
 │ intake/profile│
 └───────┬───────┘
         ▼
 ┌───────────────┐      ┌────────────────┐
 │ DNA classifier├─────►│ hypothesis gen │
 └───────┬───────┘      └───────┬────────┘
         └──────────────┬───────┘
                        ▼
                ┌───────────────┐
                │ TEPE scheduler│
                └───────┬───────┘
                        ▼
               execute / validate
                        ▼
                    ensemble
```

## how it works (TEPE)

Each hypothesis is scored as:

`score = expected_gain / expected_runtime * (1 - overfit_risk)`

After each run, TEPE updates gain/runtime/risk posteriors and re-ranks the queue. That means Prometheus reacts to *evidence*, not vibes.

## commands

- `prometheus init` — profile dataset + seed policy queue
- `prometheus run` — run policy loop for top hypotheses
- `prometheus status` — inspect ranked queue
- `prometheus blend` — blend OOF predictions

## docs

- Architecture details: `docs/architecture.md`
