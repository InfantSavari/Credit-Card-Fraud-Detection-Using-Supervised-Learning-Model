import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import joblib
import os

def evaluate_model(model, X_test, y_test, model_name, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    f = open(file,"+a")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*60}",file=f,end='\n')
    print(f"MODEL: {model_name}",file=f,end='\n')
    print(f"{'='*60}",file=f,end='\n')

    print("\nConfusion Matrix:",file=f,end='\n')
    print(confusion_matrix(y_test, y_pred),file=f,end='\n')

    print("\nClassification Report:",file=f,end='\n')
    print(classification_report(y_test, y_pred),file=f,end='\n')

    print(f"ROC-AUC Score: {roc_auc:.4f}",file=f,end='\n')
    print(f"PR-AUC Score : {pr_auc:.4f}",file=f,end='\n')
    f.close()


x_test = pd.read_parquet(r"Train_and_Test\x_test.parquet")
y_test = pd.read_parquet(r"Train_and_Test\y_test.parquet").values.ravel()

# log_mdl = joblib.load(r"Models\LogisticRegression.pkl")
# svm_mdl = joblib.load(r"Models\k-nn.pkl")
# rf_mdl= joblib.load(r"Models\randomForest.pkl")
knn_mdl = joblib.load(r"Models\k-nn.pkl")

# evaluate_model(log_mdl,x_test,y_test,"Logistics Regression",r"ouput_evaluation/logisticsRegression.txt")
# evaluate_model(svm_mdl,x_test,y_test,"Decision Tree",r"ouput_evaluation/Decision_tree.txt")
# evaluate_model(rf_mdl,x_test,y_test,"Random Forest Classification",r"ouput_evaluation/random.txt")
evaluate_model(knn_mdl,x_test,y_test,"k-nn",r"ouput_evaluation/knn.txt")









