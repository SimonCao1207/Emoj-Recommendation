import json

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.r_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=6
        )

        self.r_model.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        self.r_model = self.r_model.eval()
        self.r_model = self.r_model.to(self.device)

    def preprocessing(self, text):
        input_id = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=256, pad_to_max_length=True
        )
        attention = [float(i > 0) for i in input_id]
        input_id = torch.tensor([input_id]).to(self.device)
        attention = torch.tensor([attention]).to(self.device)
        return input_id, attention

    def predict(self, text):
        input_ids, attention_mask = self.preprocessing(text)

        with torch.no_grad():
            probabilities = F.softmax(self.r_model(input_ids, attention_mask)[0])
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            config["CLASS_NAMES"][predicted_class],
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities)),
        )


model = Model()


def get_model():
    return model
