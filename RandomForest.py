from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
x_train = pd.read_parquet(r"Train_and_Test\x_train.parquet")
y_train = pd.read_parquet(r"Train_and_Test\y_train.parquet").values.ravel()

model = Pipeline(steps=[
    ("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))])

model.fit(x_train,y_train)
joblib.dump(model,r"Models\randomForest.pkl")
