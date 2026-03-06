# 💳 Credit Card Fraud Detection

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![ggplot2](https://img.shields.io/badge/ggplot2-FF6F61?style=for-the-badge&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Balancing-brightgreen?style=for-the-badge)
![CART](https://img.shields.io/badge/CART-Decision_Tree-orange?style=for-the-badge)

A machine learning project in **R** that detects fraudulent credit card transactions using a real-world, highly imbalanced dataset. The project covers the full pipeline — exploratory data analysis, class imbalance handling with four different sampling strategies, and a CART Decision Tree classifier with and without SMOTE, evaluated via confusion matrices.

---

## 🎯 Problem Statement

Credit card fraud detection is a classic **binary classification problem** with a critical challenge: the dataset is extremely imbalanced — only **0.172% of transactions are fraudulent**. A naive model that always predicts "legitimate" achieves 99.8% accuracy but catches zero fraud. This project addresses that directly by implementing and comparing multiple resampling strategies before model training.

---

## 📊 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
*(Not included in this repo due to file size — download directly from Kaggle)*

| Property | Detail |
|---|---|
| **Total transactions** | 284,807 |
| **Fraudulent cases** | 492 (0.172%) |
| **Features** | 30 (V1–V28 via PCA + `Time` + `Amount`) |
| **Target variable** | `Class` — 1 = Fraud, 0 = Legitimate |
| **Period** | 2 days of European cardholder transactions (September 2013) |

> V1–V28 are PCA-transformed to protect cardholder confidentiality. Only `Time` and `Amount` retain their original scale.

### ⬇️ Download Instructions

1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Load it when prompted by `file.choose()` on script run

---

## 🧠 Workflow

```
Load & Inspect Data
        │
        ▼
EDA ─── Class distribution table, proportions, pie chart
        │
        ▼
Baseline ── No-model prediction (all-zero) + Confusion Matrix
        │
        ▼
Sampling (10% of data, seed = 1) + Train/Test Split (80/20)
        │
        ▼
Imbalance Handling ─┬─ Random Oversampling (ROS)
                    ├─ Random Undersampling (RUS)
                    ├─ Combined ROS + RUS
                    └─ SMOTE (K=5, target 40% fraud)
        │
        ▼
CART Decision Tree ─┬─ Trained on SMOTE-balanced data
                    └─ Trained on original (imbalanced) data
        │
        ▼
Evaluation ── Confusion Matrix, Accuracy, Precision, Recall per model
```

---

## ✨ Key Techniques

### 🔍 Exploratory Data Analysis
- Class distribution table and proportions
- Pie chart of fraud vs. legitimate transactions
- Scatter plots of V1 vs. V2 colored by class — before and after resampling

### ⚖️ Four Imbalance Handling Strategies (via `ROSE` & `smotefamily`)

| Method | Approach | Result |
|---|---|---|
| **ROS** (Random Oversampling) | Duplicates minority (fraud) samples | 50/50 split |
| **RUS** (Random Undersampling) | Removes majority (legit) samples | 50/50 split |
| **ROS + RUS Combined** | Both simultaneously | 50/50 split |
| **SMOTE** | Synthesizes new fraud samples (K=5 neighbors) | ~40/60 fraud/legit |

### 🌳 CART Decision Tree (via `rpart`)
- Trained on **SMOTE-balanced** training data
- Trained on **original imbalanced** training data
- Both evaluated on the same held-out test set
- Decision tree visualized with `rpart.plot`

---

## 📂 Project Structure

```
credit-card-frauddetection/
├── fraud_detection.R       # Full analysis script
├── README.md
└── data/
    └── creditcard.csv      # ← Download from Kaggle (not included)
```

---

## ⚙️ Getting Started

### Prerequisites

- R 4.0+
- RStudio (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/carlotarzua/credit-card-fraudetection.git
   cd credit-card-fraudetection
   ```

2. **Install required R packages** — paste this into your R console:
   ```r
   install.packages(c("caret", "dplyr", "ggplot2", "caTools", "ROSE", "smotefamily", "rpart", "rpart.plot"))
   ```

3. **Run the script**  
   Open `fraud_detection.R` in RStudio and run it. When `file.choose()` is called, navigate to your downloaded `creditcard.csv`.

---

## 📈 Results

| Model | Notes |
|---|---|
| **No-model baseline** | 99.83% accuracy — but 0% fraud recall. Confirms why accuracy alone is misleading. |
| **CART (imbalanced)** | High accuracy, poor fraud detection due to class skew |
| **CART + SMOTE** | Improved fraud recall after synthetic minority oversampling |

> 💡 The key insight: **SMOTE-balanced training significantly improves the model's ability to detect actual fraud**, at the cost of some precision — a worthwhile trade-off in a real fraud detection scenario where missing fraud is far more costly than a false alarm.

---

## 🔑 Key Takeaways

- **Accuracy is a misleading metric** when classes are heavily imbalanced — the baseline "predict all legitimate" model proved this clearly
- **SMOTE must be applied only on the training set** to prevent data leakage into evaluation
- **Visual comparison** of class distributions before/after resampling reveals how each strategy reshapes the decision boundary
- **Decision trees are interpretable** — rpart.plot makes it easy to explain model logic to non-technical stakeholders

---

## 🌱 Future Improvements

- [ ] Add Random Forest or XGBoost for performance comparison
- [ ] Tune CART hyperparameters (tree depth, min split)
- [ ] Compute full precision/recall/F1 breakdown using `caret`
- [ ] Add ROC-AUC curve comparison across sampling methods
- [ ] Build a Shiny dashboard to visualize results interactively

---

## 👩‍💻 About Me

Built by **Carlota Arzúa** — passionate about using data science and machine learning to solve real-world problems, with hands-on experience in R, Python, and applied ML workflows.

- 💼 [LinkedIn](https://www.linkedin.com/in/carlota-a-53a75b206/)
- 🌐 [Portfolio]()
- 📧 carlotaarzua@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
