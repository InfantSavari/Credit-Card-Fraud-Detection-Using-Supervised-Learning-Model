import pandas as pd 
import dask.dataframe as dd

raw_data = dd.read_csv(r"data.csv")
raw_data.to_parquet("input/")