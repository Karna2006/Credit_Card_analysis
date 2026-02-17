# Credit Card Default Prediction 🚀

Comprehensive **credit card default prediction** pipeline using Taiwan UCI dataset (25,121 samples). Predicts next-month default risk with **financially interpretable features** and **F2-score optimization** for imbalanced classes.

## 🎯 **Problem Statement**
Predict `nextmonthdefault` (0/1) for credit card holders using 6-month payment history, demographics, and financial data. **Class imbalance**: 80.9% non-default, 19.1% default.

## 📊 **Dataset**
| Feature Type | Examples | Count |
|--------------|----------|-------|
| Demographics | age, sex, marriage, education | 4 |
| Credit Limit | LIMIT_BAL | 1 |
| Payment Status | pay0-pay6 (-2 to 8) | 7 |
| Bill Amounts | Billamt1-6 | 6 |
| Payments | payamt1-6 | 6 |
| **Target** | nextmonthdefault | 1 |

**Source**: UCI ML Repository (preprocessed)

## 🏗️ **Notebook Structure**

Credit_card_analysis.ipynb
├── 📈 1. Data Loading & EDA
├── 📊 2. Descriptive Statistics
├── 🔧 3. Feature Engineering (12+ features)
├── ✂️ 4. Feature Selection (corr > 0.3)
├── 🧹 5. Preprocessing Pipeline
├── 🤖 6. Model Training & Tuning
├── 📈 7. Evaluation (F2-optimized)
├── 🔍 8. SHAP Interpretability
└── 📤 9. Submission Pipeline


## ✨ **Key Features Engineered**

| Feature | Description | Business Rationale |
|---------|-------------|-------------------|
| `CreditUtilization3M` | Avg recent bills ÷ limit | **Revolving debt risk** |
| `DelinquencySeverity` | Time-weighted payment delays | **Persistence of bad behavior** |
| `MaxDelinquencyStreak` | Consecutive late payments | **Habitual delinquency** |
| `PaymentCompliance` | Timely payments ÷ months | **Payment discipline** |
| `DebtSlope` | Bill amount trend | **Debt accumulation speed** |

## 🔬 **Methodology**

```mermaid
graph TD
    A[Raw Data] --> B[EDA & Visualizations]
    B --> C[Feature Engineering]
    C --> D[Correlation Filtering]
    D --> E[SMOTEENN Balancing]
    E --> F[XGBoost + RF + LR]
    F --> G[GridSearchCV Tuning]
    G --> H[F2 Threshold Opt]
    H --> I[SHAP Analysis]
    I --> J[Predictions]



🚀 Quick Start
bash
# 1. Clone repository
git clone https://github.com/yourusername/credit-card-default-prediction.git
cd credit-card-default-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook Credit_card_analysis.ipynb
📦 Requirements
text
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
📈 Key Results
Model	F2-Score	Recall	Precision
XGBoost	0.78	0.82	0.75
Random Forest	0.74	0.79	0.71
Logistic Regression	0.68	0.72	0.65
🔍 Feature Importance (SHAP)
text
1. DelinquencySeverity (40%)
2. MaxDelinquencyStreak (25%)
3. CreditUtilization3M (15%)
4. pay0 (12%)
5. PaymentCompliance (8%)
🎓 Perfect For
ML portfolios & interviews

Fintech risk modeling

Credit scoring systems

Imbalanced classification studies

📝 Future Improvements
 Ensemble stacking

 Time-series validation

 External economic features

 Model deployment (FastAPI)

 Time-series validation

 External economic features

 Model deployment (FastAPI)
