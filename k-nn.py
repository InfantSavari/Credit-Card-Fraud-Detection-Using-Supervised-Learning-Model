from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

x_train = pd.read_parquet(r"Train_and_Test\x_train.parquet")
y_train = pd.read_parquet(r"Train_and_Test\y_train.parquet").values.ravel()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",   
        metric="minkowski"
    ))
])

model.fit(x_train, y_train)

joblib.dump(model,r"Models\k-nn.pkl")
