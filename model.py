import os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        
    def forward(self, x):
        with torch.no_grad():
            features = self.cnn(x)
        return features    

class Decoder(nn.Module):
    
    def __init__(self, feat_size, embded_size, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embdedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(feat_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.feat_size = feat_size
        self.embded_size = embded_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, features, captions):
        embeddings  = self.embed(captions)
        inputs = torch.cat((features, embeddings), 1)
        hidden, (hs, cs) = self.rnn(inputs)
        outputs = self.fc(hidden)
        return outputs
        
    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
        
    def sample(self, features, caption, states=None, max_len=20):
        sampled = []
        for i in range(max_len):
            input = self.embed(features, caption)
            hidden, states = self.rnn(input, states)
            caption = self.fc(hidden.squeeze(1))
            pred = output.argmax(1)
            sampled.append(pred.item())
        return sampled
        
    #def beam_search(self, features, caption, states=None, max_len=20, beam_width=5):
    #    sampled = []
    #    for _ in range(beam_width):
    #        sampled.append([])
    #    hs, cs = self.init_hidden(), self.init_hidden() 
    #    for _ in range(max_len):
    #        candidates = []
    #        inputs
            