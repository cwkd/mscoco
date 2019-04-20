import os, sys

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np

from model import Encoder, Decoder
from dataloader2 import Vocabulary, MSCOCODataset, one_hot_encode, un_one_hot_encode
import pickle
import matplotlib.pyplot as plt
import nltk
#nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
        
vocab_size = len(vocab)
batch_size = 1

encoder = Encoder().to(device)
decoder = Decoder(feat_size=4096, embed_size=1000, 
                  vocab_size=vocab_size, hidden_size=300, 
                  num_layers=3).to(device)

print('Loading model')
decoder.load_state_dict(torch.load('3_model.pt'))

transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),])

valset = MSCOCODataset(root = './val2014/',
                       annFile = './annotations/captions_val2014.json',
                       transform=transform, vocab=vocab)

val_loader = data.DataLoader(valset, batch_size, shuffle=False)

def get_path_from_idx(valset, index):
    ann_id = valset.ids[index]
    img_id = valset.coco.anns[ann_id]['image_id']
    path = valset.coco.loadImgs(img_id)[0]['file_name']
    return path
    
def get_caption_from_idx(valset, index):
    ann_id = valset.ids[index]
    caption = valset.coco.anns[ann_id]['caption']
    return caption
    
outfile = open('captions.txt', 'a')
for index, (data, target) in enumerate(val_loader):        
    #print(target)
    path = get_path_from_idx(valset, index)
    caption = get_caption_from_idx(valset, index)
    
    data = data.to(device)
    
    img_feat = encoder(data).unsqueeze(0)
    start = target[0].to(device).long().unsqueeze(0)
    output = decoder.sample(img_feat, start)
    output = [vocab.idx2word[e.item()] for e in output]
    output = ' '.join(output)
    line = '{}_{}_{}'.format(path, caption, output)
    outfile.write()
    
outfile.close()