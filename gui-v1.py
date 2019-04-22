from tkinter import *
from tkinter import font
from tkinter import filedialog
from tkinter.messagebox import showinfo
from tkinter.ttk import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
#from model import Encoder, Decoder
import pickle
#from dataloader2 import Vocabulary, un_one_hot_encode

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
        
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        
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
            self.h0 = torch.zeros(self.num_layers,1,self.hidden_size).to(self.device)
            self.c0 = torch.zeros(self.num_layers,1,self.hidden_size).to(self.device)
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
            caption = caption.unsqueeze(0).to(self.device)
        return sampled

class Demonstrator(Tk):
    def __init__(self):
        super().__init__()

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=11)

        self.title("Demonstrator")
        self.geometry("1024x576")
        self.tabs = []        
        
        self.msg_frame = Frame(self)
        self.msg_frame.config(border = 5, relief = RAISED)
        self.msg_frame.pack(fill=BOTH,expand=1)
        
        welcome_msg = '\n'.join([
                '50.039 Theory and Practice of Deep Learning: The Big Project\n',
                'Loo Binn',
                'Daniel Chin',
                'Laura Ong\n',
                'Welcome! This is a GUI to demonstrate the capabilities of',
                'our image captioning model, trained on the MSCOCO Dataset.',
                'You can do 2 things with this demonstrator:\n',
                '1. In the "Predict" tab, you can choose an image from a local',
                '    directory, and the model will generate a caption for it.\n',
                '2. In the "Browse" tab, you can view the captions generated',
                '    for the images in the validation set.\n'
                ])
            
        self.msg = Label(self.msg_frame,
                         text = welcome_msg)
#        self.msg.config(justify=CENTER)
        self.msg.pack(side=TOP,padx=20,pady=20)
        
        self.close_msg_button = Button(self.msg_frame, 
                                        text="BEGIN",
                                        command=self.close_msg)
        self.close_msg_button.pack(side=TOP,padx=20,pady=20)
        
        self.notebook = Notebook(self)
        
        self.img_names = []
        directory = os.fsencode('./val2014/')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                self.img_names.append(filename)
                continue
            else:
                continue

    def add_new_tab(self, tab, name):
        self.tabs.append(tab)
        self.notebook.add(tab, text=name)
        
    def close_msg(self):
        self.msg_frame.pack_forget()
        self.notebook.pack(fill=BOTH, expand=1)


class PredictTab(Frame):
    
    def __init__(self,demonstrator):
        super().__init__(demonstrator)
        
        # Load vocabulary.
        with open('vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        vocab_size = len(self.vocab)
        
        # Create encoder, decoder and load trained weights.
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        self.encoder = Encoder().to(self.device)
        self.model = Decoder(feat_size=4096, embed_size=1000, 
                          vocab_size=vocab_size, hidden_size=300, 
                          num_layers=3).to(self.device)
        best_state_dict = torch.load(os.path.join('model.pt'),
                                     map_location=device_name)
        self.model.load_state_dict(best_state_dict)
        
        self.transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
            
        # Create functions needed for testing.
#        self.softmax = nn.LogSoftmax(dim=1)
        
        centre = Frame(self)
        centre.pack()
        
        image_frame = Frame(centre)
#        image_frame.grid(row = 0, column = 0)
        image_frame.config(border = 5, relief = RAISED)
        image_frame.pack(side=LEFT,fill=BOTH,padx=20,pady=20)
        
        self.name_holder = Label(image_frame)
        f
        self.image_holder = Label(image_frame)
#        self.image_holder.pack()
        
        self.caption_holder = Label(image_frame)
        
        self.choose_img_button = Button(image_frame, 
                                        text="Choose Image",
                                        command=self.predictForImage)
        self.choose_img_button.pack(side=BOTTOM,padx=20,pady=20)
        
        
    def getImage(self):
        
        path = filedialog.askopenfilename(filetypes=[("Image File",'*')])
        
        image_name = path[path.rfind('/')+1:]
#        print(path.rfind('/')+1)
        
        if path != '':
            im = Image.open(path).convert('RGB')
        else:
            im = None
            
        return im, image_name

    def displayChosenImage(self,im):
        img_w,img_h = im.size
        limit = 300
        if img_w >= img_h:
            resized_image = im.resize((limit,int(img_h*limit/img_w)),
                                             Image.ANTIALIAS)
        else:
            resized_image = im.resize((int(img_w*limit/img_h),limit),
                                             Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized_image)
#        print(tkimage.height())
#        print(tkimage.width())
        self.image_holder.image = tkimage
        self.image_holder.config(image=tkimage)
#        print('Displayed')
#        plt.subplot(111)
#        plt.imshow(im,cmap='gray',interpolation='none')
#        plt.xticks([])
#        plt.yticks([])
#        plt.show()

    def test(self,im):
        
        # Prepare image.
        data = self.transform(im).to(self.device).unsqueeze(0)
        
        # Prepare start of caption marker.
        idx = torch.tensor(self.vocab('<start>')).view((1,1))
        
        # Prepare caption
        caption = ''
        
        self.model.eval()
        with torch.no_grad():
            img_feat = self.encoder(data).unsqueeze(0)
            output = self.model.sample(img_feat, idx)
            output = [self.vocab.idx2word[e.item()] for e in output]
            caption = ' '.join(output)
#            features = self.encoder(data).repeat([1, 1, 1])
#            self.model.init_hidden()
#            for i in range(20): # Just in case our model never produces EOS
#                output,_ = self.model(features, idx)
##                output = self.softmax(output)
#                output = F.log_softmax(output,dim=2)
##                print(output.max())
#                idx = output.argmax().view((1,1))
#                if idx.item() == self.vocab('<end>'):
#                    break
#                else:
#                    caption += self.vocab.idx2word[idx.item()] + ' '
#            caption = caption.capitalize()
#            caption = caption[:-1] + '.'
        return caption
    
    def predictForImage(self):
        im = self.getImage()
        if im[0] != None:
            self.displayChosenImage(im[0])
            
            # Run test() on this image
            caption = self.test(im[0])
            
            self.name_holder.config(text = im[1])
            self.name_holder.pack(side=TOP)
            self.image_holder.pack(side=TOP,padx=20,pady=20)
            self.caption_holder.config(text = caption)
            self.caption_holder.pack(side=TOP)


class ImageFrame(Frame):
    
    def __init__(self,frame):
        super().__init__(frame)
        
        self.img_dir = './val2014/'
        
        self.text_holder = Frame(self)
        self.text_holder.grid(row=0,
                              column=1,
                              sticky = N+S+E+W,
                              padx = 20)
        
        self.title = Label(self.text_holder)
        self.title.config(anchor=W)
        self.title.grid(row=0,column=0,sticky=E+W,padx=5,pady=5)
        
        self.img = Label(self)
        self.img.grid(row=0,column=0,sticky = N+S+E+W)
        
        self.caption = Label(self.text_holder)
        self.caption.config(anchor=W)
#        self.caption.grid(row=1,column=0,sticky=E+W,padx=5,pady=5)
        self.caption.grid(row=1,column=0,sticky=E+W,padx=5)
        
        self.caption2 = Label(self.text_holder)
        self.caption2.config(anchor=W)
#        self.caption2.grid(row=2,column=0,sticky=E+W,padx=5,pady=5)
        self.caption2.grid(row=2,column=0,sticky=E+W,padx=5)
        
    def change(self,idx,title,caption):
        
        self.title.config(text = 'Image: '+title)
        
        img_full_path = os.path.join(self.img_dir,
                                    title)
        pil_image = Image.open(img_full_path)
        img_w,img_h = pil_image.size
        limit = 200
        if img_w >= img_h:
            resized_image = pil_image.resize((limit,int(img_h*limit/img_w)),
                                             Image.ANTIALIAS)
        else:
            resized_image = pil_image.resize((int(img_w*limit/img_h),limit),
                                             Image.ANTIALIAS)
        self.img.image = ImageTk.PhotoImage(resized_image)
        self.img.config(image = self.img.image)
        
        if len(caption) < 75:
            self.caption.config(text = caption)
            self.caption2.grid_remove()
        else:
            space = caption[:80].rfind(' ')
            self.caption.config(text = caption[:space])
            self.caption2.config(text = caption[space+1:])
            self.caption2.grid()
        
class BrowseTab(Frame):
    
#    def __init__(self,demonstrator):
#        super().__init__(demonstrator)
#        
#        self.img_names = demonstrator.img_names
#        
#        self.browse_frame = BrowseFrame(self)
#        self.browse_frame.pack(side=TOP, fill=BOTH, expand=1)
#        
#    def ppSelect(self,pp):
#        self.browse_frame.changeClass(cls)
#        self.cls_select_frame.changePage(self.browse_frame.page_number)
#        
#    def pageSelect(self,page):
#        self.browse_frame.changeClass(cls)
#        self.cls_select_frame.changePage(self.browse_frame.page_number)
#        
#    def prevPage(self):
#        if self.browse_frame.page_number > 1:
#            self.browse_frame.prevPage()
#            self.cls_select_frame.changePage(self.browse_frame.page_number)
#        
#    def nextPage(self):
#        if self.browse_frame.page_number < self.browse_frame.page_limit:
#            self.browse_frame.nextPage()
#            self.cls_select_frame.changePage(self.browse_frame.page_number)
        
        
#class BrowseFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Get image names and list of numbers as in directory.
        self.image_names = parent.img_names
        self.img_nums = {}
        
        for i in range(len(self.image_names)):
            name = self.image_names[i]
            self.img_nums[i] = name[name.rfind('_')+1:-4].lstrip('0')
        
        # Load captions from file.
        caption_file = os.path.join('captions4.txt')
        with open(caption_file,'r') as file:
            self.captions = file.readlines()
        
#        self.canv = Frame(self)
        self.canvas = Canvas(self, borderwidth=0, background="#ffffff")
#        self.sel1 = Frame(self)
#        self.sel2 = Frame(self)
#        self.selector1 = Canvas(self.sel1, borderwidth=0, background="#ffffff")
#        self.selector2 = Canvas(self.sel2, borderwidth=0, background="#ffffff")

        self.viewPort = Frame(self.canvas)
        self.viewPort.image_names = self.image_names
        
#        self.selector1_frame = Frame(self.selector1)
        self.selector1_frame = Frame(self)
        
#        self.selector2_frame = Frame(self.selector2)
        self.selector2_frame = Frame(self)
        
        self.page_number = 1
        self.pp_number = 1
        
        self.page_size = 150
        self.pp_size = 16
        self.page_limit = int(np.ceil(len(self.image_names)/self.page_size))
        self.pp_limit = int(np.ceil(self.page_limit/self.pp_size))
        
        self.cols = 5
        
        self.image_list = []
        
        for i in range(self.page_size):
            
            self.image_list.append(ImageFrame(self.viewPort))
            self.image_list[i].config(border=10,relief=SUNKEN)
#            self.image_list[i].pack(side=TOP,fill=BOTH,expand=True)
            self.image_list[i].grid(row = i,
                           sticky = N+S+E+W)
#            self.image_list[i].grid(row = int(i/self.cols),
#                           column = int(i%self.cols),
#                           sticky = N+S+E+W)
            
        self.pp_list = []
        
        for i in range(self.pp_limit):
            max_idx = min(
                    (i+1)*self.pp_size*self.page_size-1,len(self.img_nums)-1)
            self.pp_list.append(Button(
                    self.selector1_frame,
                    command = lambda idx = i: self.gotoPP(idx+1),
                    text = '{}-{}'.format(
                            self.img_nums[
                                    i*self.pp_size*self.page_size],
                            self.img_nums[
                                    max_idx])))
            self.pp_list[i].grid(row = i,sticky = N+S+E+W)
            
        self.page_list = []
        
        for i in range(self.pp_size):
            max_idx = min(
                    ((self.pp_number-1)*self.pp_size+i+1)*self.page_size-1,
                    len(self.img_nums)-1)
            self.page_list.append(Button(
                    self.selector2_frame,
                    command = lambda idx = i: self.gotoPage(idx+1),
                    text = '{}-{}'.format(
                            self.img_nums[
                                    ((self.pp_number-1)*self.pp_size+i)*self.page_size],
                            self.img_nums[max_idx])))
            self.page_list[i].grid(row = i,sticky = N+S+E+W)
            
        self.gotoPage(self.page_number)
            
        
        self.viewPort.pack(fill=BOTH,expand=True)        
        self.vsb = Scrollbar(self.canvas, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
        self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")
        self.canvas.create_window((4,4), window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                  tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.onFrameConfigure)  
        
#        self.vsb1 = Scrollbar(self.selector1, orient="vertical", command=self.selector1.yview)
#        self.selector1.configure(yscrollcommand=self.vsb1.set)
#
#        self.vsb1.pack(side="right", fill="y")
#        self.selector1.create_window((4,4), window=self.selector1_frame, anchor="nw",            #add view port frame to canvas
#                                  tags="self.selector1_frame")
#        
#        self.selector1_frame.bind("<Configure>", self.onFrameConfigure)   
#                             #bind an event whenever the size of the viewPort frame changes.
#
#        self.vsb2 = Scrollbar(self.selector2, orient="vertical", command=self.selector2.yview)
#        self.selector2.configure(yscrollcommand=self.vsb2.set)
#
#        self.vsb2.pack(side="right", fill="y")
#        self.selector2.create_window((4,4), window=self.selector2_frame, anchor="nw",
#                                  tags="self.selector2_frame")
#        
#        self.selector2_frame.bind("<Configure>", self.onFrameConfigure)
        
        self.selector1_frame.pack(side=LEFT,fill=Y)
        self.selector2_frame.pack(side=LEFT,fill=Y)
        
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
#        self.selector1.pack(side=LEFT, fill=Y, expand=True)
#        self.selector2.pack(side=LEFT, fill=Y, expand=True)
        
#        self.sel1.pack(side=LEFT,fill=BOTH,expand=True)
#        self.sel2.pack(side=LEFT,fill=BOTH,expand=True)
#        self.canv.pack(side=LEFT,fill=BOTH,expand=True)
        
    def onFrameConfigure(self, event):                                              
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))                 #whenever the size of the frame changes, alter the scroll region respectively.

    def gotoPP(self,pp):
        self.pp_number = pp
        
        last_cell = min(
                self.pp_size,
                self.page_limit-(self.pp_number-1)*self.pp_size)
        
        for i in range(self.pp_size):
            if i < last_cell:
                max_idx = min(
                        ((self.pp_number-1)*self.pp_size+i+1)*self.page_size-1,
                        len(self.img_nums)-1)
                self.page_list[i].config(
                        text = '{}-{}'.format(
                                self.img_nums[
                                        ((self.pp_number-1)*self.pp_size+i)*self.page_size],
                                self.img_nums[max_idx]))
                self.page_list[i].grid()
            else:
                self.page_list[i].grid_remove()

    def gotoPage(self,pg):
        self.page_number = pg
        last_cell = min(
                self.page_size,
                len(self.image_names)-((self.pp_number-1)*self.pp_size+(pg-1))*self.page_size)
        for i in range(self.page_size):
            if i < last_cell:
                img_idx = ((self.pp_number-1)*self.pp_size+pg-1)*self.page_size+i
                title = self.image_names[img_idx]
                caption = self.captions[img_idx]
                self.image_list[i].change(img_idx,title,caption)
                self.image_list[i].grid()
            else:
                self.image_list[i].grid_remove()
#        self.vsb.set(0,1)
        self.canvas.yview_moveto(0)

class ClassSelectFrame(Frame):
    
    def __init__(self,browse_tab):
        super().__init__(browse_tab)
        
        self.page_navi = Frame(self)
        self.page_navi.pack(side=TOP,pady=10)
        
        self.prev_page = Button(self.page_navi,
                                command = lambda:browse_tab.prevPage(),
                                text = '<')
        self.prev_page.grid(row = 0, column = 0, padx = 10)
        
        page = browse_tab.browse_frame.page_number
        self.num_images = len(browse_tab.browse_frame.image_names)
        self.page_size = browse_tab.browse_frame.page_size
        self.image_range = Label(self.page_navi,
                                 text = 'Page %d of %d - Images %d to %d'
                                 % (page,
                                    int(self.num_images/self.page_size)+1,
                                    self.page_size*(page-1)+1,
                                    min(self.page_size*page,self.num_images)))
        self.image_range.grid(row = 0, column = 1, padx = 10)
        
        self.next_page = Button(self.page_navi,
                                command = lambda:browse_tab.nextPage(),
                                text = '>')
        self.next_page.grid(row = 0, column = 2, padx = 10)
        
        self.sorted_by = Label(self,
                               text = 'Sorted by: '+self.classes[browse_tab.browse_frame.cls])
        self.sorted_by.pack(side=TOP,pady=10)
        
    def changePage(self,page):
        self.image_range.config(text = 'Page %d of %d - Images %d to %d'
                                % (page,
                                   int(self.num_images/self.page_size)+1,
                                   self.page_size*(page-1)+1,
                                   min(self.page_size*page,self.num_images)))
        self.sorted_by.config(text = 'Sorted by: '+self.classes[browse_tab.browse_frame.cls])
    

if __name__ == "__main__":
    
    demonstrator = Demonstrator()
    predict_tab = PredictTab(demonstrator)
    demonstrator.add_new_tab(predict_tab,"Predict")
    browse_tab = BrowseTab(demonstrator)
    demonstrator.add_new_tab(browse_tab,"Browse")
    demonstrator.mainloop()
    
    
    
    