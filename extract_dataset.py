import datetime as dt

import pandas as pd

now = dt.datetime.now()
df = pd.read_csv("A_Z Handwritten Data.csv", skiprows=lambda x: x % 10 != 0)
df.to_csv("az_1_10.csv", index=False, header=False)
print('cost:', dt.datetime.now() - now)
