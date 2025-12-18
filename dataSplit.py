import pandas as pd 
from sklearn.model_selection import train_test_split

x = pd.read_parquet(r"Preprocessed_data/featured_data.parquet")
y = pd.read_parquet(r"Preprocessed_data/fraud.parquet").values.ravel()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
x_train.to_parquet(r"Train_and_Test/x_train.parquet")
x_test.to_parquet(r"Train_and_Test/x_test.parquet")
y_train.to_parquet(r"Train_and_Test/y_train.parquet")
y_test.to_parquet(r"Train_and_Test/y_test.parquet")
