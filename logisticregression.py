from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

x_train = pd.read_parquet(r"Train_and_Test\x_train.parquet")
y_train = pd.read_parquet(r"Train_and_Test\y_train.parquet").values.ravel()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegressionCV(
        Cs=10,                 
        cv=5,                  
        scoring='roc_auc',     
        class_weight='balanced',
        max_iter=1000,
        n_jobs=-1,
        solver='lbfgs'
    ))
])

pipeline.fit(x_train,y_train)

joblib.dump(pipeline,r"Models/LogisticRegression.pkl")
