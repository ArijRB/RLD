import torch
import torchvision
import torch.nn as nn
import torch.functional as F


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
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

DIMRED = 5
DIMINTER = 30
DIMENTREE = 784
DIMSORTIE = 784

encodeur = Encodeur(DIMENTREE,DIMRED,DIMINTER)
decodeur = Decodeur(DIMRED,DIMSORTIE,DIMINTER)

optim_enc = torch.optim.Adam(encodeur.parameters())
optim_dec = torch.optim.Adam(decodeur.parameters())

for batch_idx, (data, target) in enumerate(train_loader):
    images = data.view(data.shape[0], -1) # bsx784
    ecart, moyenne = encodeur(images)
    bs = images.shape[0]
    random = torch.zeros(bs, DIMRED).data.normal_(0, 1)
    #torch.distribution.normal.Normal()
    #d.rsample

    z = ecart*random+moyenne
    pred = decodeur(z)

    # reconstruction loss
    print(pred.shape)
    print(images.shape)
    recon_loss = nn.BCELoss(pred, images)

    # kl divergence loss
    kl_loss = 0.5 * torch.sum(torch.exp(ecart) + moyenne**2 - 1.0 - ecart)

    # total loss
    loss = recon_loss + kl_loss
    loss.backward()
    optim_dec.step()
    optim_enc.step()