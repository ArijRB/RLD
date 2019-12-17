import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn


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


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, capacity,tailleDescription,epsilon,miniBatchSize,gamma,stepMAJ):
        self.action_space = action_space
        self.capacity = capacity
        self.RM = []
        self.Q = NN(tailleDescription,action_space.n,[14,14])
        self.Q_m = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters())
        self.epsilon = epsilon
        self.lastAction = None
        self.lastDesc = None
        self.compteur = 0
        self.miniBatchSize = miniBatchSize
        self.step = 0
        self.gamma = gamma
        self.stepMAJ = stepMAJ

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
        triplet = (self.lastDesc.numpy(),self.lastAction,reward,descriptionEtat.numpy(),done)
        if(len(self.RM)< self.capacity):
            self.RM.append(triplet)
        else:
            indice = self.compteur%self.capacity
            self.RM[indice] = triplet
            self.compteur+=1

        self.optimizer.zero_grad()
        rand_indice = np.random.randint(0,len(self.RM),self.miniBatchSize)
        print("self_RM",self.RM)
        miniBatch = np.array(self.RM)[rand_indice]
        #print("minibatch",miniBatch)
        criterion = torch.nn.SmoothL1Loss()
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
        print("x",x.shape)
        pred = self.Q(x)
        action_eff = torch.LongTensor(action_eff)
        print(pred.shape)
        print(action_eff.shape)
        pred = pred[range(self.miniBatchSize),action_eff]
        print(pred.shape)
        print(y.shape)
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

    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    agent = DQNAgent(action_space = env.action_space, capacity = 1000,tailleDescription = 6,epsilon = 0.1 ,miniBatchSize = 128,gamma = 0.9,stepMAJ=10)


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialiser le pseudo aleatoire
    episode_count = 10000
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