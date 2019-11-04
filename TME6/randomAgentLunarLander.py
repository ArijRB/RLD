import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn



class NN_V(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x

class NN_Pi(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return nn.softmax(x)

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self,tailleDesc, nbAction, nbEvent, beta=1,gamma=0.001,nbstep,lambd):
        self.V = NN_V(tailleDesc,1,[5])
        self.Pi = NN_Pi(tailleDesc,nbAction,[5])
        self.nbAction = nbAction
        self.nbEvent = nbEvent
        self.beta = beta
        self.gamma = gamma
        self.nbstep = nbstep
        self.lambd = lambd
        self.compteur = 0
        self.list_e_a_r_d = []
        self.old_obs = None
        self.old_action = None

    def act(self, observation, reward, done):
        if(self.old_obs != None):
            self.list_e_a.append([self.old_obs,self.old_action,reward,done])

        a = np.argmax(self.Pi(observation))
        self.old_obs = observation
        self.old_action = a

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            optimisation()
            self.compteur = 0
            self.list_e_a_r = []
        
        return a
    
    def optimisation():
        list_avantage = calc_avantage()
        self.label_V = []
        t_r=0
        for t in range(len(self.list_e_a_r_d)):
            self.label_V.append(np.power(self.gamma,t_r)*self.list_e_a_r_d[t][2])
            t_r+=1
            if(self.list_e_a_r_d[t][3]):
                t_r=0
        #Optimiser le réseaux V avec les differents labels

        #Optimiser le réseaux Pi
        for _ in range(self.nbstep):



    def calc_avantage():
        av = []
        for indice in range(0,len(self.list_e_a_r_d),-1):
            r = self.list_e_a_r[indice][2]
            obs = self.list_e_a_r[indice][0]
            if(not(done)):
                calc = r+self.gamma * self.V(self.list_e_a_r[indice+1][0]) - self.V(obs) + self.gamma*self.lambd*av[-1]
            else:
                calc = r - self.V(obs)
            av.append(calc)
        return av





if __name__ == '__main__':


    env = gym.make('LunarLander-v2')

    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    agent = DQNAgent(action_space = env.action_space, capacity = 1000,tailleDescription = 8,epsilon = 0.1 ,miniBatchSize = 128,gamma = 0.9,stepMAJ=10)

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    env._max_episode_steps = 200
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 10 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
