from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

import pandas as pd
from torch import nn

import spacy

class MessageClassifer():
    def __init__(self, MODEL_NAME="intent-model_1"):
        label_df = pd.read_csv("labels.txt", sep="\t")
        self.labels = dict(zip(label_df["Labels"], label_df["Dialogue Act"]))
        n_classes = len(label_df["Labels"])
        label_index = dict(zip(label_df["Labels"],range(n_classes)))
        self.label_index_inv = {v: k for k, v in label_index.items()}

        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

        nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp

    def text_to_sents(self, text):
        doc = self.nlp(text)

        return [str(doc[sent.start: sent.end]) for sent in doc.sents]

    def get_tag(self, logits):
        preds = nn.functional.softmax(logits, dim=-1)

        score = torch.max(preds).data.item()
        label = torch.argmax(preds).data.item()
        return (self.labels[self.label_index_inv[label]], score)

    def predict(self, input_text):

        inputs = self.tokenizer(input_text, return_tensors="pt") #.to("cpu")
        outputs = self.model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        return self.get_tag(logits)