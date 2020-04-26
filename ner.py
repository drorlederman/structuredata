import pandas as pd
import numpy as np
from tqdm import tqdm, trangeimport torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score

data = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
data = data.fillna(method='ffill')print("Number of sentences: ", len(data.groupby(['Sentence #'])))words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)data.head(n=10)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return Nonegetter = SentenceGetter(data)


sent = getter.get_next()
print(sent)

sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]

labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}

MAX_LEN = 75
bs = 32device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

