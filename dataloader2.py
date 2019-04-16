import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

class MSCOCODataset(Dataset.CocoCaptions):
    def __init__(self,root,annFile,transform, vocab):
#         super().__init__(root,annFile, transform)
        self.root = root
        self.transform = transform
        
        self.img_path = []
        directory = os.fsencode(root)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"): 
                self.img_path.extend(os.fsdecode(os.path.join(directory, file)))# attach name of the file
                continue
            else:
                continue
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        
    def __getitem__(self, index):
#         image, target = super().__getitem__(self, index)
        ann_id = self.ids[index]
        print(ann_id)
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
             # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        print(tokens)
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
#         target = torch.Tensor(caption)
        return image, caption #caption returns as a lst of indiced words from vocab i.e. [1, 4, 12, 27, 14, 4, 15, 7, 28, 29, 2]
        #image is returned as a tensor. I disabled target just in case. 
    def __len__(self):
        return len(self.ids)

def one_hot_encode(arr, n_labels):#inputs a list, and the number of words in the vocab list
    arr = np.asarray(arr)
    # Initialize the the encoded array
    t =arr.shape[0]
    one_hot = np.zeros((t, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot #returns a numpy array
def un_one_hot_encode(one_hot)
    lst = [np.where(r==1)[0][0] for r in one_hot]
    return lst #retursn a list from numpy array
def main():
    vocab = build_vocab('./annotations_trainval2014/annotations/captions_train2014.json', 11)
    vocab_path = 'vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    cap = MSCOCODataset(root = './train_images/train2014/',
                            annFile = './annotations_trainval2014/annotations/captions_train2014.json',
                            transform=transforms.ToTensor(), vocab=vocab)

    print('Number of samples: ', len(cap))
    img, target = cap[3] # load 4th sample

    print("Image Size: ", img.size())
    print(target)

    len(one_hot_encode(target, len(vocab))[0]) 
    un_one_hot_encode(one_hot_encode(target, len(vocab)))
