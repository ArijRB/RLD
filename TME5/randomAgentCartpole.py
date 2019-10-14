import matplotlib

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



class A3C(object):
    """The world's simplest agent!"""
    def __init__(self, action_space,tailleDescription, tMax,gamma):
        self.action_space = action_space
        self.Pi = NN2(tailleDescription,action_space.n,[24,24])
        self.V = NN1(tailleDescription,1,[24,24])
        self.tMax = tMax
        self.t = 0
        self.tstart = 0
        self.saveR = []
        self.saveObs = []
        self.action = []
        self.gamma = gamma
        self.optimPi = torch.optim.Adam(self.Pi.parameters(),lr=1e-3)
        self.optimV = torch.optim.Adam(self.V.parameters(),lr=1e-3)

    def act(self, observation, reward, done):
        descriptionEtat = torch.Tensor(observation)
        act = np.argmax(self.Pi(descriptionEtat).detach().numpy())
        if(not(done) or self.t-self.tstart==self.tMax):
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
            for i in range(self.t-1,self.tstart,-1):
                R = self.saveR[i] + self.gamma * R
                new = torch.log(self.Pi(self.saveObs[i])[self.action[i]])*(R-self.V(self.saveObs[i]).detach())
                new.backward()
                new2 = torch.pow(R-self.V(self.saveObs[i]),2)
                new2.backward()
            self.optimPi.step()
            self.optimV.step()
            self.tstart = self.t
        return act








if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    agent = A3C(action_space = env.action_space,tailleDescription = 4, tMax=50,gamma = 0.99)
    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 10000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
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
