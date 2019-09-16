import numpy as np

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

#Baseline random
reward_cum = 0
for k in range(len(data_contexte)):
    annonce_choisi = np.random.randint(0,9)
    reward_cum+=data_annonce[k][annonce_choisi]
print("Baseline random")
print(reward_cum)

#Baseline staticBest
arr = np.array(list(data_annonce.values()))
annonce_static = np.argmax(np.sum(arr,axis=0))
reward_cum = 0
for k in range(len(data_contexte)):
    reward_cum+=data_annonce[k][annonce_static]
print("Baseline staticBest")
print(reward_cum)

#Baseline optimale
reward_cum = 0
for k in range(len(data_contexte)):
    reward_cum+=np.max(data_annonce[k])
print("Baseline optimale")
print(reward_cum)

#Algo UCB

reward_cum = 0
#Initialisation moyenne empirique
moyenne_empirique = np.zeros(10)
tirer_annonce = np.zeros(10)
for k in range(10):
    moyenne_empirique[k] = data_annonce[k][k]
    tirer_annonce[k]+=1
Bt = np.zeros(10)
for k in range(10,len(data_contexte)):
    for i in range(10):
        Bt[i] = moyenne_empirique[i] + np.sqrt( ( 2*np.log(k) ) / (tirer_annonce[i]) )
    choix = np.argmax(Bt)
    reward_cum+=data_annonce[k][choix]
    moyenne_empirique[choix] = (moyenne_empirique[choix]*tirer_annonce[choix]+data_annonce[k][choix])/(tirer_annonce[choix]+1)
    tirer_annonce[choix]+=1
print("Algo UCB")
print(reward_cum)

#Algo LinUCB
reward_cum = 0
dim_contexte = 5
nb_annonceur = 10
A = np.array([np.identity(dim_contexte) for _ in range(nb_annonceur)])
B = np.zeros((nb_annonceur,dim_contexte))
teta = np.random.rand(nb_annonceur,dim_contexte)
alpha = 0.1
for k in range(len(data_contexte)):
    P_t = np.zeros(nb_annonceur)
    for i in range(nb_annonceur):
        teta[i] = np.matmul(np.linalg.inv(A[i]),B[i])
        vecteur_t1 = np.transpose([[t] for t in teta[i]])
        vecteur_t2 = data_contexte[k]
        t3 = data_contexte[k].reshape((dim_contexte,1)).T
        membre2 = np.matmul(t3,np.linalg.inv(A[i]))
        membre3 = np.matmul(membre2,data_contexte[k])
        P_t[i] = np.matmul(vecteur_t1,vecteur_t2)[0] + alpha*np.sqrt(membre3[0])
    choisi=np.argmax(P_t)
    A[choisi]=A[choisi]+data_contexte[k].T*data_contexte[k]
    B[choisi]=B[choisi]+data_annonce[k][choisi]*data_contexte[k]
    reward_cum+=data_annonce[k][choisi]

print(reward_cum)