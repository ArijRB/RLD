import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import time
import matplotlib.pyplot as plt


class PolicyAgent(object):
    def __init__(self, action_space,states,gamma,nb_iter,epsilon):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = gamma
        self.states = states
        self.nb_iter = nb_iter
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        return self.Pi[self.states[gridworld.GridworldEnv.state2str(observation)]]

    def fit(self,statedic, mdp):
        diff_value = []
        self.V = np.random.random(len(statedic))
        self.Pi = np.random.randint(0,self.action_space.n,len(statedic))
        nb_iter = self.nb_iter
        epsilon = self.epsilon
        nb_etape_totale = 0
        nb_etape_value = 0
        for iter_en_cours in range(nb_iter):
            #Mise a jours de V
            for _ in range(nb_iter):
                ancien_V = np.copy(self.V)
                for cle_s,valeur_s in statedic.items():
                    if(cle_s not in mdp.keys()):continue
                    somme = 0
                    action_a_faire = self.Pi[valeur_s]
                    list_succ = mdp[cle_s][action_a_faire]
                    for noeud in list_succ:
                        somme +=noeud[0]*( noeud[2] + self.gamma*self.V[valeur_s])
                    self.V[valeur_s] = somme
                nb_etape_value+=1
                if(ancien_V is not None):
                    diff_value.append(np.linalg.norm(ancien_V-self.V))
                if(ancien_V is not None and np.linalg.norm(ancien_V-self.V) < epsilon): break
            #Mise a jours de Pi
            ancien_Pi = np.copy(self.Pi)
            for cle_s,valeur_s in statedic.items():
                if(cle_s not in mdp.keys()): continue
                li = np.zeros(self.action_space.n)
                for a in range(self.action_space.n):
                    somme = 0
                    list_succ = mdp[cle_s][a]
                    for noeud in list_succ:
                        entier_noeud = statedic[noeud[1]]
                        somme +=noeud[0]*( noeud[2] + self.gamma*self.V[entier_noeud])
                    li[a] = somme
                self.Pi[valeur_s] = np.argmax(li)
            nb_etape_totale+=1
            if(ancien_Pi is not None and np.array_equal(self.Pi, ancien_Pi) ):
                print("La politique n'a pas changé entre les deux itérations donc on suppose qu'elle est devenue stable")
                break
        return nb_etape_totale,nb_etape_value,diff_value

class ValueAgent(object):
    def __init__(self, action_space,states,gamma,nb_iter,epsilon):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = gamma
        self.states = states
        self.nb_iter = nb_iter
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        return self.Pi[self.states[gridworld.GridworldEnv.state2str(observation)]]

    def fit(self,statedic, mdp):
        diff_value = []
        self.V = np.random.random(len(statedic))
        self.Pi = np.random.randint(0,self.action_space.n,len(statedic))
        nb_iter = self.nb_iter
        epsilon = self.epsilon

        nb_etape_value = 0
        #Mise a jours de V
        for _ in range(nb_iter):
            ancien_V = np.copy(self.V)
            for cle_s,valeur_s in statedic.items():
                if(cle_s not in mdp.keys()):continue
                maxi = -99999
                for action_a_faire in range(self.action_space.n):
                    somme = 0
                    list_succ = mdp[cle_s][action_a_faire]
                    for noeud in list_succ:
                        somme +=noeud[0]*( noeud[2] + self.gamma*self.V[valeur_s])
                    if(somme> maxi):
                        maxi=somme
                self.V[valeur_s] = maxi
            nb_etape_value+=1
            if(ancien_V is not None):
                    diff_value.append(np.linalg.norm(ancien_V-self.V))
            if(ancien_V is not None and np.linalg.norm(ancien_V-self.V) < epsilon):
                print(np.linalg.norm(ancien_V-self.V))
                break
        
        #Mise a jours de Pi
        for cle_s,valeur_s in statedic.items():
            if(cle_s not in mdp.keys()): continue
            li = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                somme = 0
                list_succ = mdp[cle_s][a]
                for noeud in list_succ:
                    entier_noeud = statedic[noeud[1]]
                    somme +=noeud[0]*( noeud[2] + self.gamma*self.V[entier_noeud])
                li[a] = somme
            self.Pi[valeur_s] = np.argmax(li)
            
        return nb_etape_value,diff_value


def test_policy_iteration(num_plan):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan"+str(num_plan)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    # Execution avec un Agent
    agent = PolicyAgent(env.action_space,statedic,gamma=0.99,nb_iter=1000,epsilon=1e-8)
    tmps_debut = time.time()
    nb_etape_totale,nb_etape_value,diff_value = agent.fit(statedic,mdp)
    print("Résultat policy iteration :")
    print("L'entrainement à pris",time.time()-tmps_debut," secondes pour converger")
    print("Il a fallu",nb_etape_totale,"étapes de mise a jours de politique et",nb_etape_value," étapes de mise a jours de la value function")
    plt.clf()
    plt.title("Courbe de distance entre V_t et V_t+1 pour Policy iteration")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Distance")
    plt.plot(diff_value)
    #plt.show()
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 100
    reward = 0
    done = False
    rsum = 0
    somme_all_episode = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                somme_all_episode+=rsum
                #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    print("La reward moyenne sommer sur les episodes est de ",somme_all_episode/episode_count)
    env.close()



def test_value_iteration(num_carte):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan"+str(num_carte)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
   
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    # Execution avec un Agent
    agent = ValueAgent(env.action_space,statedic,gamma=0.99,nb_iter=100000,epsilon=1e-10)
    tmps_debut = time.time()
    nb_etape_value,diff_value = agent.fit(statedic,mdp)
    print("Résultat Value iteration :")
    print("L'entrainement à pris",time.time()-tmps_debut," secondes pour converger")
    print("Il a fallu",nb_etape_value," étapes de mise a jours de la value function")
    plt.clf()
    plt.title("Courbe de distance entre V_t et V_t+1 pour Value iteration")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Distance")
    plt.plot(diff_value)
    #plt.show()
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 100
    reward = 0
    done = False
    rsum = 0
    somme_all_episode = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                somme_all_episode+=rsum
                #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    print("La reward moyenne sommer sur les episodes est de ",somme_all_episode/episode_count)
    env.close()


if __name__ == '__main__':
    #test_policy_iteration(1)
    test_value_iteration(1)

    