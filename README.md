# Titanic Survival Prediction — End-to-End ML Pipeline

A mini-project implementing a complete, deployable `Pipeline` — combining data cleaning, imputation, encoding, scaling, and model training into a single chained object — to predict passenger survival on the Titanic dataset.

## 📌 Overview

This project moves beyond individual feature engineering practice notebooks into a full workflow: raw data goes in, a trained `DecisionTreeClassifier` and predictions come out — all through one `Pipeline` object. It also documents a real debugging experience with `ColumnTransformer` column-reordering, a common pitfall when chaining multiple transformers.

## 📂 Files

| File | Description |
|---|---|
| `pipeline-implementation-on-titanic-data.ipynb` | Main notebook — full pipeline from raw data to accuracy score |
| `train.csv` | Titanic training dataset (891 rows) — passenger details and survival outcome |

## 🧪 Dataset

The classic [Titanic dataset](https://www.kaggle.com/c/titanic) — passenger details (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) with the target `Survived`. `PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped as they don't contribute to prediction.

## 🔍 What the Notebook Covers

1. **Data cleaning** — drops irrelevant columns, inspects missing values (`Age`, `Embarked`).
2. **Train/test split**.
3. **Combined preprocessing** — a single `ColumnTransformer` handles imputation (`Age`, `Embarked`) and One-Hot Encoding (`Sex`, `Embarked`) together, using column names rather than positional indices.
4. **Scaling** — `MinMaxScaler` normalizes the numeric features.
5. **Model training** — a `DecisionTreeClassifier` is trained as the final pipeline step.
6. **Evaluation** — predictions are generated on the test set and scored with `accuracy_score`.

## 📊 Result

| Model | Accuracy |
|---|---|
| Pipeline (DecisionTreeClassifier, default params) | ~0.765 |

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scikit-learn
```

## ▶️ How to Run

1. Make sure `train.csv` is in the same directory as the notebook (update the path in the notebook if needed).
2. Open the notebook in Jupyter:
   ```bash
   jupyter notebook pipeline-implementation-on-titanic-data.ipynb
   ```
3. Run all cells in order.

## 📚 Key Learning: Debugging a ColumnTransformer Bug

An earlier version of this pipeline chained **three separate `ColumnTransformer`s** (one each for imputation, encoding, and scaling), using positional column indices. This broke — each `ColumnTransformer`'s output reorders columns, so indices that were correct for the original data pointed to the wrong columns after the first transform, causing a `ValueError`.

**Fix:** combining imputation and encoding into a **single** `ColumnTransformer`, referencing columns by **name** instead of position, avoided the reordering issue entirely.

## 🔮 Possible Improvements

- Hyperparameter tuning (`max_depth`, `min_samples_split`) via `GridSearchCV`
- Trying other models (Random Forest, Logistic Regression)
- Feature engineering (e.g. extracting titles from names, family size from `SibSp` + `Parch`)

---

*Part of my Machine Learning practice journey — learning core feature engineering and pipeline concepts hands-on.*
