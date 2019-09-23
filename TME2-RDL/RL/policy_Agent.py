import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class PolicyAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = 0.99

    def act(self, observation, reward, done):
        return self.Pi[observation]
    
    def fit(self,mdp):
        self.V = np.random.random(len(mdp))
        self.Pi = np.random.randint(0,self.action_space.n,len(mdp))
        mdp = list(mdp.values())
        nb_iter = 100
        for _ in range(nb_iter):
            for i in range(len(self.V)):
                somme = 0
                for j in range(len(mdp)):
                    somme += mdp[i][self.Pi[state[i]]][j][0]*( mdp[i][self.Pi[i]][j][2] + self.gamma*self.V[i])
                self.V[i] = somme
            for i in range(len(self.V)):
                li = np.zeros(len(self.action_space))
                for a in range(len(self.action_space)):
                    somme = 0
                    for j in range(len(state)):
                        somme += mdp[i][a][j][0]*( mdp[i][a][j][2] + self.gamma*self.V[j])
                    li[a] = somme
                self.Pi[i] = np.argmax(li)





if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    print("*************************")
    print(transitions[0])
    # Execution avec un Agent
    agent = PolicyAgent(env.action_space)
    agent.fit(mdp)

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
        #env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
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