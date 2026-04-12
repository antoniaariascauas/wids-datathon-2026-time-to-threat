# WiDS Datathon 2026 — Time-to-Threat Prediction

Submission for the [Women in Data Science (WiDS) Global Datathon 2026](https://www.widsdatathon.org/). The challenge: predict the probability that a critical event occurs within 12, 24, 48, and 72 hours, framed as a survival/hazard problem.

## Problem

Given patient-level features, predict cumulative probabilities P(event <= t) for four time horizons. The evaluation metric is mean Brier score across all horizons — rewarding well-calibrated probability estimates, not just ranking.

## Approach

### Modeling Strategy
- **Hazard-based framing**: Instead of training 4 independent classifiers, I model discrete interval hazards (probability of event in each interval, given survival to that point) and reconstruct cumulative probabilities. This enforces monotonicity by construction.
- **LightGBM** as the primary model with monotonic constraints on hazard features
- **Calibration**: Platt scaling and isotonic regression to improve probability calibration post-training
- **Ensemble**: Blending across constraint modes (core/full/off), calibration methods, and sharpening strategies

### Post-Processing Pipeline
1. **Monotonic enforcement** — isotonic regression across horizons ensures P(t1) <= P(t2) <= P(t3) <= P(t4)
2. **Sharpening** — temperature scaling and gamma transforms push predictions toward {0, 1} to optimize Brier score
3. **Zero-floor / One-ceil** — snap extreme probabilities and propagate consistency across horizons
4. **Ladder quantization** — optional discretization to empirical probability levels

### Models Compared
| Model | Type |
|-------|------|
| LightGBM (hazard) | Discrete hazard with interval-level modeling |
| LightGBM (direct) | Independent per-horizon binary classifiers |
| HistGradientBoosting | scikit-learn native gradient boosting |
| Random Forest | Bagging ensemble |
| Extra Trees | Randomized splits ensemble |
| Logistic Regression | Linear baseline with standard scaling |

## Project Structure

```
├── wids_time_to_threat_full_pipeline.py   # Full pipeline (2,077 lines)
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── metaData.csv
```

## How to Run

```bash
pip install -r requirements.txt

# Default run (hazard LightGBM + isotonic calibration + sharpening)
python wids_time_to_threat_full_pipeline.py

# With super-sharpening and ladder quantization
python wids_time_to_threat_full_pipeline.py --super-sharp --ladder "0,0.4074,0.7143,1"

# Specific models only
python wids_time_to_threat_full_pipeline.py --models hazard_lgb direct_lgb
```

## Key Design Decisions

1. **Hazard framing over direct classification** — Modeling interval hazards naturally produces monotonic cumulative probabilities and shares information across horizons. Direct classifiers treat each horizon independently, missing the sequential structure.

2. **Brier score optimization** — Since the metric rewards calibration (not just AUC), post-processing focused on probability accuracy: isotonic calibration, temperature scaling, and sharpening to avoid the "mushy middle" problem.

3. **Exhaustive variant search** — The pipeline evaluates all combinations of constraint modes, calibration methods, blending strategies, and sharpening parameters, then selects the best by cross-validated Brier score.

## Tech Stack

Python, LightGBM, scikit-learn, pandas, NumPy
