# %%
import start  # noqa

import pandas as pd

from dpp4nlp.dpp4nlp import DPP4NLP
from text_encoder.models import RoBERTaModel, MXBai

# %%
analysis = DPP4NLP(model=MXBai())

# %%
dataframe = pd.read_csv("data/News_Category_Dataset_v3_50.csv")
dataframe.head(3000)

# %%
x_dataframe = dataframe[["headline", "short_description"]]
x_dataframe.head()
# %%
y = dataframe["category"].tolist()
y[:3000]
# %%
analysis.dp = True
analysis.get_representative_embeddings(x_dataframe, sample=3000, y=y)

# %%
