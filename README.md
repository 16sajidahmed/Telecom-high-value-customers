# Predicting High-Value Customers in Telecom

A data-driven retention and upsell strategy for a telecom operator ("Company A"), built for the GCI World 2026 Final Assignment.

## Problem

Company A wants to know who its most valuable customers are, whether it is keeping them, and where the next wave of high-value customers will come from. This project answers four questions:

1. How is revenue distributed across customers?
2. Are the most valuable customers being retained?
3. What service quality issues affect high-value customers?
4. Is there an untapped segment trending toward high value?

High-value customers are defined as the top 20% by average monthly revenue (`avgrev` ≥ $76.86). "Rising Stars" are low-value customers whose usage is trending upward (`change_mou > 0`).

## Data

100,000 customers, 100 features after merging `Client.csv` and `Record.csv` on `Customer_ID`. The raw data is not included in this repository. Place both files in a `data/` folder and set `PATH` in the notebook.

## Approach

1. Merge and explore the data (revenue concentration, churn, Rising Stars, usage vs revenue)
2. Remove leaked features, impute missing values, one-hot encode categoricals (217 features)
3. 80/20 train/test split, with Logistic Regression and Random Forest as baselines
4. Optuna tuning of Logistic Regression with 3-fold stratified CV
5. Domain features: `calls_per_month`, `revenue_per_call`, `completion_rate`
6. Quantified business impact and verification of the proposal

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.907 | 0.943 |
| Random Forest | 0.897 | 0.934 |
| Logistic Regression (Optuna) | 0.906 | 0.943 |

Optuna did not improve on the default configuration, so default hyperparameters were already near-optimal for this dataset. Engineered features (`calls_per_month`, `completion_rate`) changed ROC-AUC by +0.0006, which is not material. A revenue-per-call feature was excluded because it is built from `avgrev`, the column that defines the label.

Business findings:

- 20.0% of customers (20,001) are high-value; their average revenue is $112.22/month.
- High-value churn is 48.3%, about 9,669 customers.
- 35,558 Rising Stars (44.4% of low-value customers) earn $44.32/month on average and show +$10.12 average monthly revenue change versus -$6.50 for stagnant low-value customers.
- Base-case opportunity (10% Rising Star conversion plus 10% high-value churn reduction): about $4.2M per year.

## Limitations

- The model separates high-value customers well (average predicted probability 0.666) but does not separate Rising Stars from stagnant low-value customers (0.082 vs 0.085). It identifies who is valuable today, not who is about to become valuable; Rising Star targeting relies on the `change_mou` rule.
- Revenue impact figures depend on assumed conversion and churn-reduction rates, not measured uplift.

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Stack

Python, pandas, NumPy, scikit-learn, Optuna, matplotlib, seaborn
