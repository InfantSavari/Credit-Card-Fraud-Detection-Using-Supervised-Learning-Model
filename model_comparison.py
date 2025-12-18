import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import joblib

X_test = pd.read_parquet("Train_and_Test/x_test.parquet")
y_test = pd.read_parquet("Train_and_Test/y_test.parquet").values.ravel()

models = {
    "Logistic Regression": joblib.load("models/LogisticRegression.pkl"),
    "Random Forest": joblib.load("models/randomForest.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "KNN": joblib.load("models/k-nn.pkl"),
}



plt.figure(figsize=(8,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("Precision–Recall Curve Comparison (Fraud Detection).png")
plt.show()
