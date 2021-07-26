from typing import Dict

import torch
import random
from torch.nn import Embedding
import os
import numpy as np


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()

        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.dropout = dropout
        self.dropout = torch.nn.Dropout(p=0.25)
        self.bidirectional = bidirectional
        self.num_class = num_class

        # self.rnn = torch.nn.RNN(embeddings.shape[1], hidden_size, num_layers, bidirectional = self.bidirectional)
        # self.fc1 = torch.nn.Linear(hidden_size*2, num_class) 

        self.lstm = torch.nn.LSTM(embeddings.shape[1], hidden_size, batch_first=True, bidirectional = self.bidirectional)
        self.fc = torch.nn.Linear(hidden_size*2, num_class)
        # TODO: model architecture

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else args.device)
        
        # Look up the embedding
        x = self.embed(batch)
        # x = self.dropout(x)

        out, (ht, ct) = self.lstm(x)
        
        y = self.fc(out[:, -1, :])
        # y = torch.softmax(y, dim = 1)
        return y

        # TODO: implement model forward
        raise NotImplementedError
