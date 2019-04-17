#!/usr/bin/env python
# coding: utf-8
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

torch.manual_seed(42)
np.random.seed(42)

def train(max_epochs=10, encoder= None, model=None, train_loader=None, val_loader=None, 
          optimizer=None, loss_fn=None, vocab_size=None, verbose=False, print_every=2000):
    train_losses, val_losses, val_accuracies = [], [], []
    batch_size = train_loader.batch_size
    device = torch.device('cuda')
    model.to(device)
    softmax = nn.LogSoftmax(dim=1)
    loss = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            gt = target.copy()
            target = torch.cat(target[:-1]).unsqueeze(0)
            gt = torch.cat(gt[1:]).unsqueeze(0)
            data, target, gt = data.to(device), target.to(device), gt.to(device)
            features = encoder(data).unsqueeze(0)
            features = features.repeat([1, gt.shape[1], 1])
            #print(features.shape)
            output, (hidden, cs) = model(features, target)
            output = softmax(output)
            #optimizer.zero_grad()
            #target, _ = nn.utils.rnn.pad_packed_sequence(target, batch_first=True)
            #data, _ = nn.utils.rnn.pad_packed_sequence(data, batch_first=True)
            #print(data.shape, target.shape)
            loss = 0
            for b in range(batch_size):
                for e in range(len(gt[b])):
                    l = loss_fn(output[b,e].unsqueeze(0), gt[b,e].long().unsqueeze(0))
                    loss += l
            loss/batch_size
            #pred = output.argmax(dim=2, keepdim=True) # get the index of the max log-probability
            #print(pred.shape, target.shape)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            if verbose and batch_idx*batch_size % print_every < batch_size:
                print('Batch Number: {}\t\t[{}/{}\t({:.3f}%)]\tLoss: {:.6f}'.format(
                        batch_idx + 1, (batch_idx +1)* train_loader.batch_size, len(train_loader),
                        100. * batch_idx / (len(train_loader)/train_loader.batch_size), loss.item()))
            loss.backward()
            optimizer.step()
            #if batch_idx*batch_size % 5000 < batch_size and batch_idx!=0:
            #    sample(model, temp)
            
        train_losses.append(loss.item())
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch + 1, (batch_idx+1) * train_loader.batch_size, len(train_loader),
            100. * batch_idx / (len(train_loader)/train_loader.batch_size), loss.item()))
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for data, target in val_loader:
                gt = target.copy()
                target = torch.cat(target[:-1]).unsqueeze(0)
                gt = torch.cat(gt[1:]).unsqueeze(0)
                data, target, gt = data.to(device), target.to(device), gt.to(device)
                features = encoder(data).unsqueeze(0)
                features = features.repeat([1, gt.shape[1], 1])
                output, (hidden, cs) = model(features, target)
                #target, _ = nn.utils.rnn.pad_packed_sequence(target, batch_first=True)
                #data, _ = nn.utils.rnn.pad_packed_sequence(data, batch_first=True)
                loss = 0
                for b in range(batch_size):
                    for e in range(len(gt[b])):
                        l = loss_fn(output[b,e].unsqueeze(0), gt[b,e].long().unsqueeze(0))
                        loss += l
                        total += 1
                batch_loss = loss.item()
                val_loss += batch_loss
                pred = output.argmax(dim=2, keepdim=True) # get the index of the max log-probability
                #print(pred, target)
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            #if val_loss < best_loss[0]:
            #    print('\nSaving model parameters in epoch {}'.format(epoch+1))
            #    torch.save(model.state_dict(), "./temp_model.pt")
            #    best_loss[0] = val_loss
            #    save_flag = True

            val_acc = 100. * correct / total
            val_accuracies.append(val_acc)

            print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss, correct, total,
                val_acc))
            
        #sample(model, temp)
    return train_losses, val_losses, val_accuracies

def main(max_epochs=20, batch_size=1, lr=1e-4):
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    vocab_size = len(vocab)
    #print(vocab.word2idx['<end>'])
    
    print('Vocab Size: {}'.format(vocab_size))
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),])
    
    
    trainset = MSCOCODataset(root = './train2014/',
                            annFile = './annotations/captions_train2014.json',
                            transform=transform, vocab=vocab)
    print('Number of training samples: ', len(trainset))
    
    valset = MSCOCODataset(root = './val2014/',
                            annFile = './annotations/captions_val2014.json',
                            transform=transform, vocab=vocab)

    
    print('Number of validation samples: ', len(valset))
    
    device = torch.device('cuda')
    encoder = Encoder().to(device)
    decoder = Decoder(feat_size=4096, embed_size=1000, 
                      vocab_size=vocab_size, hidden_size=300, 
                      num_layers=3).to(device)
    
    train_loader = data.DataLoader(trainset, batch_size, shuffle=False)
    val_loader = data.DataLoader(valset, batch_size, shuffle=False)
    optimizer = optim.RMSprop(decoder.parameters(), lr)
    loss_fn = nn.NLLLoss()
    
    train_losses, val_losses, val_accs = train(max_epochs, encoder, decoder, train_loader, val_loader, 
                                               optimizer, loss_fn, vocab_size, verbose=True)
    
    best_model_wts = decoder.state_dict()
    torch.save(best_model_wts,'./model.pt')

    epochs = np.arange(1, max_epochs+1)
    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_losses, epochs, val_losses)
    plt.legend(['Training', 'Validation'],loc='upper right')
    plt.title('Training and Validation Loss per epoch')
    plt.show()

    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, val_accs)
    plt.legend(['Validation'],loc='upper right')
    plt.title('Validation Accuracy per epoch')
    plt.show()
    
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    val_accs = np.asarray(val_accs)
    np.save('./train_losses.npy',train_losses)
    np.save('./val_losses.npy',train_losses)
    np.save('./val_accs.npy',train_losses)

main()



