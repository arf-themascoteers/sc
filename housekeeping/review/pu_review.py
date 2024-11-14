import pandas as pd


df = pd.read_csv("../../data/paviaU.csv")
unique_values_with_counts = df['class'].value_counts()
print(unique_values_with_counts)
distinct_count = df['class'].nunique()
print(distinct_count)
print(len(df.columns))
print(len(df))