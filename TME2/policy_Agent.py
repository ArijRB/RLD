import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class PolicyAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space,states):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = 0.999
        self.states = states

    def act(self, observation, reward, done):
        return self.Pi[self.states[gridworld.GridworldEnv.state2str(observation)]]

    def fit(self,statedic, mdp):
        self.V = np.random.random(len(statedic))
        self.Pi = np.random.randint(0,self.action_space.n,len(statedic))
        nb_iter = 100000
        epsilon = 0.001
        for _ in range(nb_iter):
            ancien_V = self.V
            for cle_s,valeur_s in statedic.items():
                if(cle_s not in mdp.keys()):continue
                somme = 0
                action_a_faire = self.Pi[valeur_s]
                list_succ = mdp[cle_s][action_a_faire]
                for noeud in list_succ:
                    somme +=noeud[0]*( noeud[2] + self.gamma*self.V[valeur_s])
                self.V[valeur_s] = somme
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
            if(np.linalg.norm(ancien_V-self.V) < epsilon): break


if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print(statedic.keys())
    print(mdp.keys())

    print(list(mdp.keys())[0])
    print(list(mdp.values())[0])

    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat

    # Execution avec un Agent
    agent = PolicyAgent(env.action_space,statedic)
    agent.fit(statedic,mdp)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
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
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
