import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class NN(nn.Module):
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


class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, capacity,tailleDescription,miniBatchSize,gamma,stepMAJ,exp_replay=True):
        self.action_space = action_space
        self.capacity = capacity
        self.RM = []
        self.Q = NN(tailleDescription,action_space.n,[24,24])
        self.Q_m = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr=1e-3)
        self.epsilon = 1
        self.lastAction = None
        self.lastDesc = None
        self.compteur = 0
        self.miniBatchSize = miniBatchSize
        self.step = 0
        self.gamma = gamma
        self.stepMAJ = stepMAJ
        self.exp_replay = exp_replay

    def act(self, observation, reward, done):
        descriptionEtat = torch.Tensor(observation)
        if(np.random.random()<self.epsilon):
            action = self.action_space.sample()
        else:
            pred = self.Q_m(descriptionEtat)
            pred = pred.detach().numpy()
            maxi = np.max(pred)
            action = np.random.choice(np.where(pred == maxi)[0],1)[0]
        if(self.lastAction == None):
            self.lastAction = action
            self.lastDesc = descriptionEtat
            return action
        self.epsilon -= 0.001
        triplet = (self.lastDesc.numpy(),self.lastAction,reward,descriptionEtat.numpy(),done)
        if(len(self.RM)< self.capacity):
            self.RM.append(triplet)
        else:
            indice = self.compteur%self.capacity
            self.RM[indice] = triplet
            self.compteur+=1

        self.optimizer.zero_grad()
        criterion = torch.nn.SmoothL1Loss()
        if(self.exp_replay):
            rand_indice = np.random.randint(0,len(self.RM),self.miniBatchSize)
            miniBatch = np.array(self.RM)[rand_indice]
            x = torch.Tensor([m[0] for m in miniBatch])
            y = []
            action_eff = [m[1] for m in miniBatch]
            for m in miniBatch:
                if(m[4]):
                    y.append(m[2])
                else:
                    maxi = np.max(self.Q_m(torch.Tensor(m[3])).detach().numpy())
                    y.append(m[2]+self.gamma*maxi)
            y = torch.Tensor(y)
            pred = self.Q(x)
            action_eff = torch.LongTensor(action_eff)
            pred = pred[range(self.miniBatchSize),action_eff]
            loss = criterion(pred,y)
            loss.backward()
        else:
            rand_indice = np.array([len(self.RM)-1])
            miniBatch = np.array(self.RM)[rand_indice]
            x = torch.Tensor([m[0] for m in miniBatch])
            y = []
            action_eff = [m[1] for m in miniBatch]
            for m in miniBatch:
                if(m[4]):
                    y.append(m[2])
                else:
                    maxi = np.max(self.Q_m(torch.Tensor(m[3])).detach().numpy())
                    y.append(m[2]+self.gamma*maxi)
            y = torch.Tensor(y)
            pred = self.Q(x)
            action_eff = torch.LongTensor(action_eff)
            #pred = pred[range(self.miniBatchSize),action_eff]
            loss = criterion(pred,y)
            loss.backward()

        self.optimizer.step()

        if(self.step > self.stepMAJ):
            self.Q_m = copy.deepcopy(self.Q)
            self.step = 0
        self.step+=1
        self.lastAction = action
        self.lastDesc = descriptionEtat
        return action



if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    #Réseaux sans target network <=> stepMaj =0
    #Réseaux avec target network <=> stepMaj =10
    agent = DQNAgent(action_space = env.action_space, capacity = 1000,tailleDescription = 4 ,miniBatchSize = 64,gamma = 0.9,stepMAJ=10,exp_replay=True)
    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 250
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    all_reward = []

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
                all_reward.append(rsum)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    
    print("done")
    env.close()
    print("Itération max",np.argmax(all_reward))
    print("Le max est de",np.max(all_reward))
    plt.clf()
    plt.plot(all_reward)
    plt.show()
