import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class NN1(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN1, self).__init__()
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

class NN2(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN2, self).__init__()
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
        x = torch.nn.functional.softmax(x)
        return x



class A2C(object):
    """The world's simplest agent!"""
    def __init__(self, action_space,tailleDescription,gamma, pasMaj,inSize,outSize):
        self.action_space = action_space
        self.Pi = NN2(inSize,outSize,[30,30])
        self.V = NN1(inSize,1,[30,30])
        self.t = 0
        self.tstart = 0
        self.saveR = []
        self.saveObs = []
        self.action = []
        self.gamma = gamma
        self.optimPi = torch.optim.Adam(self.Pi.parameters(),lr=1e-3)
        self.optimV = torch.optim.Adam(self.V.parameters(),lr=1e-3)
        self.pasMaj = pasMaj

    def act(self, observation, reward, done):
        observation = self.getFeatures(observation)
        observation = observation.reshape(-1)
        #print(observation.shape)
        descriptionEtat = torch.Tensor(observation)
        #print(self.Pi(descriptionEtat).detach())
        act = torch.distributions.categorical.Categorical(self.Pi(descriptionEtat).detach())
        act = act.sample().numpy()
        #print(act)
        if(not(done)) : #and  not(self.t-self.tstart==self.tMax)):
            self.t +=1
            self.saveObs.append(descriptionEtat.detach())
            self.saveR.append(reward)
            self.action.append(act)
            return act
        else:
            if(done):
                R = 0
            else:
                R = self.V(descriptionEtat).detach()
            for i in range(len(self.saveR)-1,-1,-1):
                R = self.saveR[i] + self.gamma * R
                #print()
                new = -torch.log(self.Pi(self.saveObs[i])[self.action[i]])*(R-self.V(self.saveObs[i]).detach())
                new.backward()
                new2 = torch.pow(R-self.V(self.saveObs[i]),2)
                new2.backward()
            if(self.t%self.pasMaj == 0):
                self.optimV.step()

            self.optimPi.step()
            self.saveObs = []
            self.saveR = []
            self.action = []
            self.t = 0
        return act

    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])    
        return state.reshape(1,-1)

class FeaturesExtractor(object):
    def __init__(self,outSize):
        super().__init__()
        self.outSize=outSize*3
    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1,-1)





if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
      # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    
    env.seed(0)  # Initialiser le pseudo aleatoire
    inSize = 108   
    outSize = env.action_space.n  
    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    agent = A2C(action_space = env.action_space,tailleDescription = 108,gamma = 0.99, pasMaj = 1000,inSize=inSize,outSize=outSize)
  
    episode_count = 10000
    reward = 0
    done = False
    all_reward = []
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
                all_reward.append(rsum)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    plt.plot(all_reward)
    plt.show()