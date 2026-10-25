# BT4012 Competition
## Usage
1. Place Data Files in the data folder (See repository structure for project directory structure)
2. Run notebook in the modelling folder (gemini_modelling.ipynb)

## Repository Structure
```
.
├── data/
│   ├── train.csv                         # Transaction features and labels for steps 1–35
│   ├── test.csv                          # Transaction features for steps 36–49 (Kaggle Test)
│   ├── txs_edgelist.csv                  # Directed transaction graph edges (txId1 -> txId2)
│   └── sample_submission.csv             # Template submission file
│
├── notebooks/
│   └── gemini_modelling.ipynb           # Primary modeling & experiment notebook
│
└── submissions/
```

## Model Architecture
```
  Raw Transaction Data & Directed Edges (txs_edgelist)
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
Structural Graph Aggregations     Time-Step Standardization
(Degrees & 1-Hop Neighbors)         (Z-score per time_step)
         └────────────────┬────────────────┘
                          │
                          ▼
           BiDirectional GraphSAGE Model
        (5-Fold OOF & OOT Rank-Normalized)
                          │
                          ▼
               Feature Matrix (251 Features)
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
5-Seed LightGBM     5-Seed XGBoost      5-Seed CatBoost
      │                   │                   │
      └───────────────────┼───────────────────┘
                          ▼
               Percentile Rank Blending
                          │
                          ▼
             Final Fraud Predictions (AUC)
```

## Validation Strategy

To prevent data leakage and accurately simulate production deployment, the pipeline uses a **Strict Out-of-Time (OOT)** temporal validation scheme[cite: 1, 2]:

```
+-------------------------------------------------------------------+
| Time-Steps 1 - 27  | Time-Steps 28 - 35   | Time-Steps 36 - 49    |
|   Training Set     |   OOT Validation Set |   Final Test Set      |
|  (24,923 samples)  |     (6,312 samples)  |   (15,329 samples)    |
+-------------------------------------------------------------------+
```

* **Leakage-Free Feature Generation**: GNN predictions for training splits are generated using 5-Fold Stratified Cross-Validation (`StratifiedKFold`)[cite: 1, 2].
* **OOT Evaluation**: Models are validated strictly on future time steps (28–35) before retraining on all historical data (1–35) for test set inference (36–49)[cite: 1, 2].
