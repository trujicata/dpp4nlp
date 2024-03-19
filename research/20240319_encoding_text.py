# %%
import start

from transformers import RobertaModel, RobertaTokenizer

# %%
roberta_model = RobertaModel.from_pretrained("roberta-base")

# %%
text = "This is a test sentence"
# tokenize
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids

# %%
# encode
tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
embeddings = roberta_model(**tokenized_text).last_hidden_state  # type: ignore

pooling = "cls"

if pooling == "cls":
    pooled_output = embeddings[:, 0, :]
elif pooling == "mean":
    pooled_output = embeddings.mean(dim=1)

pooled_output  # torch.Size([1, 768])
# %%
# Save the model
import torch

torch.save(roberta_model, "roberta_model.pt")

# %%
# Save tokenizer as well
tokenizer.save_pretrained("roberta_tokenizer")

# %%
# Now, load again the model and tokenizer
roberta_model = torch.load("text_encoder/roberta/weights/roberta_model.pt")

# %%
# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(
    "text_encoder/roberta/weights/roberta_tokenizer"
)

# %%
from text_encoder.roberta.model import RoBERTaModel

roberta = RoBERTaModel()

# %%
