from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import joblib

x_train = pd.read_parquet(r"Train_and_Test\x_train.parquet")
y_train = pd.read_parquet(r"Train_and_Test\y_train.parquet").values.ravel()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear",          
        class_weight="balanced",  
        probability=True,         
        random_state=42
    ))
])

model.fit(x_train, y_train)

joblib.dump(model,r"Models\svm.pkl")