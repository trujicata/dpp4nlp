import os
import torch
from transformers import RobertaModel, RobertaTokenizer


class RoBERTaModel:
    def __init__(self, model_name: str = "roberta-base"):

        if not os.path.exists(
            f"text_encoder/roberta/weights/{model_name}.pt"
        ) or not os.path.exists(f"text_encoder/roberta/weights/{model_name}_tokenizer"):
            self.model = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

            os.makedirs("text_encoder/roberta/weights", exist_ok=True)

            torch.save(self.model, f"text_encoder/roberta/weights/{model_name}.pt")
            self.tokenizer.save_pretrained(
                f"text_encoder/roberta/weights/{model_name}_tokenizer"
            )
        else:
            self.model = torch.load(f"text_encoder/roberta/weights/{model_name}.pt")
            self.tokenizer = RobertaTokenizer.from_pretrained(
                f"text_encoder/roberta/weights/{model_name}_tokenizer"
            )

    def encode(self, text: str, pooling: str = "cls"):
        tokenized_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        embeddings = self.model(**tokenized_text).last_hidden_state
        if pooling == "cls":
            pooled_output = embeddings[:, 0, :]
        elif pooling == "mean":
            pooled_output = embeddings.mean(dim=1)
        return pooled_output
