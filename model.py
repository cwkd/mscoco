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
    
    def __init__(self, feat_size, embed_size, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(feat_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, features, captions):
        embeddings  = self.embed(captions)
        #print(features.shape, captions.shape, embeddings.shape)
        inputs = torch.cat((features, embeddings), 2)
        #print(inputs.shape)
        hidden, (hs, cs) = self.rnn(inputs,(self.h0,self.c0))
        outputs = self.fc(hidden)
        return outputs, (hs, cs)
        
    def init_hidden(self,hc=None):
        if hc == None:
            self.h0 = torch.zeros(self.num_layers,1,self.hidden_size).to(torch.device('cuda'))
            self.c0 = torch.zeros(self.num_layers,1,self.hidden_size).to(torch.device('cuda'))
        else:
            self.h0,self.c0 = hc
        
    def sample(self, features, caption, states=None, max_len=20):
        sampled = []
        for i in range(max_len):
            embedded = self.embed(caption)
            #print(features.shape, caption.shape, embedded.shape)
            input = torch.cat((features, embedded), 2)
            hidden, states = self.rnn(input, states)
            output = self.fc(hidden.squeeze(1))
            #probs = F.softmax(output, dim=1).squeeze(0).detach().cpu().numpy()
            #pred = np.random.choice(range(len(probs)), 1,p=probs)
            #caption = torch.Tensor(pred).long()
            caption = output.argmax(1)
            
            #print(caption)
            #sampled.append(caption)
            sampled.append(caption.cpu())
            if caption.item() == 2: #EOS
                break
            caption = caption.unsqueeze(0).cuda()
        return sampled
        
    #def beam_search(self, features, caption, states=None, max_len=20, beam_width=5):
    #    sampled = []
    #    for _ in range(beam_width):
    #        sampled.append([])
    #    hs, cs = self.init_hidden(), self.init_hidden() 
    #    for _ in range(max_len):
    #        candidates = []
    #        inputs
            