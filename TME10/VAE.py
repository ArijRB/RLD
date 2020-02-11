import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from PIL import Image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import time
import torch.distributions as tdist

class Encodeur(nn.Module):
    def __init__(self,dimEntree,dimReduit,dimInter):
        super(Encodeur,self).__init__()
        self.inter = nn.Linear(dimEntree,dimInter)
        self.moyenne = nn.Linear(dimInter,dimReduit)
        self.ecart_type = nn.Linear(dimInter,dimReduit)
        self.dimRed = dimReduit

    def forward(self,x):
        x = self.inter(x)
        x = nn.functional.relu(x)
        moyenne = self.moyenne(x)
        ecart = self.ecart_type(x)
        return ecart,moyenne


class Decodeur(nn.Module):
    def __init__(self,dimRed,dimSortie,dimInter):
        super(Decodeur,self).__init__()
        self.linear = nn.Linear(dimRed,dimInter)
        self.sortie = nn.Linear(dimInter,dimSortie)

    def forward(self,z):
        x = self.linear(z)
        x = nn.functional.relu(x)
        x = self.sortie(x)
        x = nn.functional.sigmoid(x)
        return x

batch_size_train = 32
batch_size_test = 32

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size_test, shuffle=True)


def random_image(dec,epoch,batch_idx):
  random = alea.rsample()
  z = random
  pred = dec(z)
  image_genere = pred.reshape(1,28,28)
  echantillon = image_genere[0]
  img = echantillon.data
  plt.imshow(img, cmap='gray')
  plt.savefig("echantillon_"+str(epoch)+"_"+str(batch_idx)+".png")

DIMRED = 100
DIMINTER = 200
DIMENTREE = 784
DIMSORTIE = 784
NUM_EPOCH = 100

encodeur = Encodeur(DIMENTREE,DIMRED,DIMINTER)
decodeur = Decodeur(DIMRED,DIMSORTIE,DIMINTER)

optim_enc = torch.optim.Adam(encodeur.parameters())
optim_dec = torch.optim.Adam(decodeur.parameters())

alea = tdist.Normal(torch.zeros(DIMRED), torch.ones(DIMRED))
for epoch in range(NUM_EPOCH):
  for batch_idx, (data, target) in enumerate(train_loader):
      images = data.view(data.shape[0], -1) # bsx784
      ecart, moyenne = encodeur(images)
      bs = images.shape[0]
      #random = torch.zeros(bs, DIMRED).data.normal_(0, 1)
      random = alea.rsample()
      #torch.distribution.normal.Normal()
      #d.rsample

      z = ecart*random+moyenne
      pred = decodeur(z)

      #Affichage échantillon
      if(batch_idx%10000==0):
        random_image(decodeur,epoch,batch_idx)
        """
        image_genere = pred.reshape(bs,28,28)
        echantillon = image_genere[0]
        im = Image.fromarray(np.uint8(cm.gist_earth(echantillon.detach().numpy())*255))
        newsize = (280, 280) 
        im = im.resize(newsize) 
        #im.show()
        img = echantillon.data
        plt.imshow(img, cmap='gray')
        plt.savefig("echantillon_"+str(epoch)+"_"+str(batch_idx)+".png")
        #plt.show()
        """

      # reconstruction loss
      bce = nn.BCELoss()
      recon_loss = bce(pred, images)

      # kl divergence loss
      #kl_loss = F.kl_div()
      kl_loss = -0.5 * torch.sum(1 + ecart - moyenne.pow(2) - ecart.exp())
      #kl_loss = 0.5 * torch.sum(torch.exp(ecart) + moyenne**2 - 1.0 - ecart)

      # total loss
      optim_dec.zero_grad()
      optim_enc.zero_grad()
      loss = recon_loss + kl_loss
      loss.backward()
      optim_dec.step()
      optim_enc.step()

      