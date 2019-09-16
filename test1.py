import numpy as np
import time
import matplotlib.pyplot as plt

fichier_data = open("data.txt")

all_lines = fichier_data.readlines()

data_contexte = dict()
data_annonce = dict()
for line in all_lines:
    tab = line.split(":")
    id = int(tab[0])
    contexte = np.array(tab[1].split(";"),dtype=np.float64)
    taux_annonce = np.array(tab[2].split(";"),dtype=np.float64)
    data_contexte[id]=contexte
    data_annonce[id]=taux_annonce

#Baseline optimale
reward_cum = 0
all_reward_cum_opti = []
for k in range(len(data_contexte)):
    reward_cum+=np.max(data_annonce[k])
    all_reward_cum_opti.append(reward_cum)
print("Baseline optimale")
print(reward_cum)
plt.plot(all_reward_cum_opti,label="Baseline optimale")
all_regret_opti = np.array(all_reward_cum_opti) - np.array(all_reward_cum_opti)

#Baseline random
reward_cum = 0
all_reward_cum_random = []
for k in range(len(data_contexte)):
    annonce_choisi = np.random.randint(0,9)
    reward_cum+=data_annonce[k][annonce_choisi]
    all_reward_cum_random.append(reward_cum)
print("Baseline random")
print(reward_cum)
plt.plot(all_reward_cum_random,label="Baseline random")
all_regret_random = np.array(all_reward_cum_random) - np.array(all_reward_cum_opti)

#Baseline staticBest
arr = np.array(list(data_annonce.values()))
annonce_static = np.argmax(np.sum(arr,axis=0))
reward_cum = 0
all_reward_cum_static = []
for k in range(len(data_contexte)):
    reward_cum+=data_annonce[k][annonce_static]
    all_reward_cum_static.append(reward_cum)
print("Baseline staticBest")
print(reward_cum)
plt.plot(all_reward_cum_static,label="Baseline staticBest")
all_regret_static = np.array(all_reward_cum_static) - np.array(all_reward_cum_opti)

#Algo UCB
reward_cum = 0
all_reward_cum_UCB = []
#Initialisation moyenne empirique
moyenne_empirique = np.zeros(10)
tirer_annonce = np.zeros(10)
for k in range(10):
    moyenne_empirique[k] = data_annonce[k][k]
    tirer_annonce[k]+=1
    reward_cum+=data_annonce[k][k]
    all_reward_cum_UCB.append(reward_cum)
Bt = np.zeros(10)
for k in range(10,len(data_contexte)):
    for i in range(10):
        Bt[i] = moyenne_empirique[i] + np.sqrt( ( 2*np.log(k) ) / (tirer_annonce[i]) )
    choix = np.argmax(Bt)
    reward_cum+=data_annonce[k][choix]
    all_reward_cum_UCB.append(reward_cum)
    moyenne_empirique[choix] = (moyenne_empirique[choix]*tirer_annonce[choix]+data_annonce[k][choix])/(tirer_annonce[choix]+1)
    tirer_annonce[choix]+=1
print("Algo UCB")
print(reward_cum)
plt.plot(all_reward_cum_UCB,label="UCB")
all_regret_UCB = np.array(all_reward_cum_UCB) - np.array(all_reward_cum_opti)

#Algo LinUCB
reward_cum = 0
all_reward_cum_LinUCB = []
dim_contexte = 5
nb_annonceur = 10
A = np.array([np.identity(dim_contexte) for _ in range(nb_annonceur)])
B = np.zeros((nb_annonceur,dim_contexte))
alpha = 0.01
for k in range(len(data_contexte)):
    P_t = np.zeros(nb_annonceur)
    t3 = data_contexte[k].reshape((dim_contexte,1)).T
    vecteur_t2 = data_contexte[k]
    teta = np.zeros((nb_annonceur,dim_contexte))
    for i in range(nb_annonceur):
        teta[i] = np.matmul(np.linalg.inv(A[i]),B[i])
        vecteur_t1 = teta[i].reshape((dim_contexte,1)).T
        membre2 = np.matmul(t3,np.linalg.inv(A[i]))
        membre3 = np.matmul(membre2,data_contexte[k])
        P_t[i] = np.matmul(vecteur_t1,vecteur_t2)[0] + alpha*np.sqrt(membre3[0])
    choisi=np.argmax(P_t)
    A[choisi]=A[choisi]+t3*data_contexte[k]
    B[choisi]=B[choisi]+data_annonce[k][choisi]*data_contexte[k][choisi]
    reward_cum+=data_annonce[k][choisi]
    all_reward_cum_LinUCB.append(reward_cum)
print("Algo LinUCB")
print(reward_cum)
plt.plot(all_reward_cum_LinUCB,label="LinUCB")
all_regret_LinUCB = np.array(all_reward_cum_LinUCB) - np.array(all_reward_cum_opti)
plt.legend()
plt.title("Reward cummulé")
#plt.show()

plt.savefig("reward_cummule.png")
plt.clf()

plt.plot(all_regret_opti,label="regret opti")
plt.plot(all_regret_random,label="regret random")
plt.plot(all_regret_static,label="regret static")
plt.plot(all_regret_UCB,label="regret UCB")
plt.plot(all_regret_LinUCB,label="regret LinUCB")
plt.title("Regret")
plt.legend()
#plt.show()

plt.savefig("regret.png")
