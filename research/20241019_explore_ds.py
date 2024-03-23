# %%
import start

import pandas as pd

# %%
# Load the data from data/News_Category_Dataset_v3.json

ds = pd.read_csv("data/News_Category_Dataset_v3_50.csv")
ds.head()

# %%
# Save in a csv file
ds.to_csv("data/News_Category_Dataset_v3.csv", index=False)
# %%
# Now, create a new dataset with only 50 rows

ds_50 = ds.head(15000)
ds_50.head()
# %%
# Save in a csv file
ds_50.to_csv("data/News_Category_Dataset_v3_50.csv", index=False)
# %%
ds = ds.iloc[:20].loc[:, ["category", "headline", "short_description"]]
ds.head()

# %%
