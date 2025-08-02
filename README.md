# Credit Card Fraud Detection 🧠💳

This project uses machine learning techniques to detect fraudulent transactions in credit card data. It focuses on identifying anomalies using classification models trained on real-world, anonymized datasets.

## 📌 Features

- Detects fraudulent transactions using ML models
- Imbalanced dataset handling
- Data preprocessing and feature scaling
- Model evaluation with precision, recall, and F1-score
- Visualizations to understand fraud patterns

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud cases**: 492 (~0.17%)

## 🧠 Algorithms Used

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost / LightGBM (optional)
- Isolation Forest / One-Class SVM (for anomaly detection)

## 🧰 Libraries Used

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Imbalanced-learn (for SMOTE or under-sampling)

## 📁 Project Structure
```
credit-card-fraud-detection
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks for training & testing
├── models/ # Saved ML models (optional)
├── fraud_detection.py # Main script
├── requirements.txt # Dependencies
├── README.md # Project description
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

# install dependencies
pip install -r requirements.txt

# Run the model script
python fraud_detection.py

# 📈 Results
Achieved accuracy of ~99.7%

F1-score for fraud class: ~0.89 (after balancing)

Precision-Recall tradeoff handled using ROC and PR curves

# 🙋‍♂️ Author
Venkat Reddy
GitHub: venkatreddy7569

# 📃 License
This project is open-source for learning and research purposes. Commercial use requires permission.

