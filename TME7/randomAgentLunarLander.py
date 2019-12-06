import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import random



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


class DDPGAgent(object):
    #DDPG SANS TARGET NETWORK
    def __init__(self,tailleDesc, low, high, number_of_update=3,batch_size=10 ,freq_update=20, gamma=0.001):
        self.Q = NN(tailleDesc+1,1,[5])
        self.mu = NN(tailleDesc,1,[5])
        self.freq_update = freq_update
        self.number_of_update = number_of_update
        self.batch_size = batch_size
        self.gamma = gamma
        self.compteur = 0
        self.list_e_a_r_d = []
        self.old_obs = None
        self.old_action = None
        self.low = low
        self.high = high

        self.criterion = torch.nn.MSELoss()
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters())
        self.optimizer_mu = torch.optim.Adam(self.mu.parameters())

    def act(self, observation, reward, done):
        bruit = np.random.normal()
        observation = torch.tensor(observation).float()
        action_pred = self.mu(observation)[0]+bruit
        action = torch.clamp(action_pred,self.low,self.high).item()
        if(self.old_obs is not None):
            mem = (self.old_obs,self.old_action,reward,observation,done)
            self.list_e_a_r_d.append(mem)  
        if(self.compteur > self.freq_update):
            for _ in range(self.number_of_update):
                B = random.choices(self.list_e_a_r_d, k=self.batch_size)
                list_target =  []
                entree_deuxieme_Q = []
                entree_update_policy = []
                for el in B:
                    s_t, a, r, s_tp1, d = el
                    action_stp1 = self.mu(s_tp1)
                    entree_Q = torch.cat((s_tp1,action_stp1))
                    one_y = r + self.gamma*(1-d)*self.Q(entree_Q).detach()
                    list_target.append(one_y)
                    one_entree = torch.cat((s_t,torch.tensor([a])))
                    entree_deuxieme_Q.append(one_entree)
                    one_policy = torch.cat((s_t,self.mu(s_t)))
                    entree_update_policy.append(one_policy)
                target = torch.tensor(list_target)
                entree = torch.stack(entree_deuxieme_Q)
                pred = self.Q(entree)
                loss = self.criterion(pred,entree)
                loss.backward()
                self.optimizer_Q.step()
                self.optimizer_Q.zero_grad()
                update_policy = torch.stack(entree_update_policy)
                sortie = -self.Q(update_policy).sum()
                sortie.backward()
                self.optimizer_mu.step()
                self.optimizer_mu.zero_grad()
                self.optimizer_Q.zero_grad()
        self.old_obs = observation
        self.old_action = action
        self.compteur+=1
        return [action]
    





if __name__ == '__main__':


    env = gym.make('MountainCarContinuous-v0')

    # Enregistrement de l'Agent
    #agent = RandomAgent(env.action_space)
    agent = DDPGAgent(2, -100, 100)

    outdir = 'MountainCarContinuous-v0/results'
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
