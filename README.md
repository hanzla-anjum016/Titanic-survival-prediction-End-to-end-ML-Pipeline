# Titanic Survival Prediction — End-to-End ML Pipeline

A mini-project implementing a complete, deployable `Pipeline` — combining data cleaning, imputation, skewness reduction, encoding, and scaling — then comparing two classifiers (Decision Tree vs Logistic Regression) to predict passenger survival on the Titanic dataset.

## 📌 Overview

This project builds a full ML workflow: raw data goes in, a trained model and predictions come out — all through a single `Pipeline` object. Along the way, it explores skewness in the data, fixes it with log transformation, and compares how two different algorithms respond to the same preprocessing.

## 📂 Files

| File | Description |
|---|---|
| `titanic-survival-prediction-end-to-end-ml-pipeline.ipynb` | Main notebook — full pipeline from raw data to model comparison |
| `train.csv` | Titanic training dataset (891 rows) — passenger details and survival outcome |

## 🧪 Dataset

The classic [Titanic dataset](https://www.kaggle.com/c/titanic) — passenger details (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) with the target `Survived`. `PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped as they don't contribute to prediction.

## 🔍 What the Notebook Covers

1. **Data cleaning** — drops irrelevant columns, inspects missing values (`Age`, `Embarked`).
2. **Skewness check** — KDE and QQ plots reveal `Fare` (and `Age`) are positively skewed.
3. **Train/test split**.
4. **Combined preprocessing** — a single `ColumnTransformer` handles:
   - `Age`: imputation → `log1p` transform (inside a sub-`Pipeline`, so both steps run in the correct order)
   - `Embarked`: imputation → One-Hot Encoding (also a sub-`Pipeline`)
   - `Sex`: One-Hot Encoding
   - `Fare`: `log1p` transform (to reduce skew)
5. **Scaling** — `MinMaxScaler` normalizes the numeric features.
6. **Model comparison** — two full pipelines are built and trained: one with `DecisionTreeClassifier`, one with `LogisticRegression`.
7. **Evaluation** — both models are scored on the test set with `accuracy_score`.

## 📊 Results

| Model | Accuracy |
|---|---|
| Decision Tree Classifier | ~78.2% |
| Logistic Regression | ~81.6% |

Logistic Regression outperformed the Decision Tree here. This also makes the `MinMaxScaler` step more meaningful than it would be for a tree-based model alone — scaling has little effect on a Decision Tree's splits, but it directly helps a gradient-based model like Logistic Regression converge properly and treat features fairly.

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## ▶️ How to Run

1. Make sure `train.csv` is in the same directory as the notebook (update the path in the notebook if needed).
2. Open the notebook in Jupyter:
   ```bash
   jupyter notebook titanic-survival-prediction-end-to-end-ml-pipeline.ipynb
   ```
3. Run all cells in order.

## 📚 Key Learnings

- **Sub-pipelines inside `ColumnTransformer`**: when a column needs more than one sequential step (e.g. impute *then* transform), it must be wrapped in its own `Pipeline` — `ColumnTransformer` entries otherwise run independently and in parallel on the original data, not on each other's output.
- **Reducing skew with `log1p`**: applied to `Age` and `Fare` to pull in their long right tails and make the distributions closer to normal, using `log1p` instead of `log` to safely handle zero values.
- **Scaling's impact depends on the algorithm**: preprocessing choices aren't one-size-fits-all — `MinMaxScaler` doesn't change a Decision Tree's decisions (trees are scale-invariant), but it matters for Logistic Regression, which is sensitive to feature scale.

## 🔮 Possible Improvements

- Hyperparameter tuning (`GridSearchCV`) for both models
- Trying additional models (Random Forest, SVM)
- Feature engineering (e.g. extracting titles from names, family size from `SibSp` + `Parch`)

---

*Part of my Machine Learning practice journey — learning core feature engineering and pipeline concepts hands-on.*
