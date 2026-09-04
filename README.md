# Credit Card Fraud Detection — Neural Network with Full Model Diagnostics

A neural network built to detect fraudulent credit card transactions, with a strong focus on **model selection and evaluation** rather than just getting a model to run. The dataset is highly imbalanced , so this project is really about diagnosing and fixing the problems that come with that — not just training a network.

## Problem Statement

Credit card fraud is rare but costly. The goal here is to build a model that actually catches fraud, while understanding and controlling the trade-off between catching fraud (recall) and avoiding false alarms (precision).

## Dataset

- **Source:** [Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/mlg-ulb/creditcardfraud) — anonymized European card transactions from September 2013, collected by Worldline and the ULB Machine Learning Group.

## Approach

1. **EDA** — class balance, correlation of each feature with `Class`, outlier check on `Amount` (fraud transactions turned out to be smaller in extreme value but higher on average than normal ones — an interesting, non-obvious finding)
2. **Preprocessing** — stratified train/val/test split (70/15/15) to preserve the fraud ratio in every split; scaled `Time` and `Amount` only (V1–V28 are already scaled from PCA)
3. **Baseline NN** — simple 2-hidden-layer network (16→8→1)
4. **Bias/variance diagnosis** — training vs validation loss curves
5. **Class imbalance handling** — class weights (`class_weight='balanced'`)
6. **Threshold tuning** — precision-recall trade-off explored across thresholds 0.5–0.995, tuned on the validation set, evaluated once on the test set
7. **Architecture comparison** — a larger network (64→32→16) tested to see if more capacity helps
8. **L2 regularization** — used to fix overfitting found in the larger network
9. **Error analysis** — inspected individual missed fraud cases and their predicted probabilities

## Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Small NN + class weights, tuned threshold (0.995) | **0.81** | **0.77** | **0.79** |
| Big NN + L2 regularization, tuned threshold (0.995) | 0.80 | 0.77 | 0.79 |

## Key Findings

- **Accuracy is meaningless here.** A 99%+ accuracy model can still miss 1 in 4 fraud cases — precision, recall, and F1 (focused on the fraud class) are what matter.
- **Class weighting alone isn't "better," it's a trade-off.** It pushed recall from 72% to 89%, but precision collapsed to 13%. Threshold tuning was needed to bring precision back up while keeping most of the recall gain.
- **Bigger isn't automatically better.** A larger architecture overfit badly (F1 dropped to 0.27) and only matched the smaller model's performance (F1 = 0.79) after adding L2 regularization — it never beat the simpler model.
- **Error analysis showed two distinct failure types:** threshold-driven near-misses (the model was suspicious, ~95% confident, but under the strict 0.995 cutoff) and genuine confusion (the model gave some real fraud cases a probability as low as 10–48%). The first is a tunable trade-off; the second suggests the anonymized PCA features may be limiting what the model can learn, and richer features or ensemble methods (NN + tree-based models) could help.
