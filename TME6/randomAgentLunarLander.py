import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import copy



class NN_V(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN_V, self).__init__()
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
        super(NN_Pi, self).__init__()
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
        return torch.nn.functional.softmax(x)


class PPOAgent(object):
    """The world's simplest agent!"""
    def __init__(self,tailleDesc, nbAction, nbEvent, nbstep,lambd, beta=1,gamma=0.99,sigma=0.03):
        self.V = NN_V(tailleDesc,1,[5])
        self.optimV = torch.optim.Adam(self.V.parameters())
        self.Pi = NN_Pi(tailleDesc,nbAction,[5])
        self.optimPi = torch.optim.Adam(self.Pi.parameters())
        self.Pi_old = NN_Pi(tailleDesc,nbAction,[5])
        self.nbAction = nbAction
        self.nbEvent = nbEvent
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.nbstep = nbstep
        self.lambd = lambd
        self.compteur = 0
        self.list_e_a_r_d = []
        self.old_obs = None
        self.old_action = None
        self.old_distrib = None

    def act(self, observation, reward, done):
        observation = observation
        if(self.old_obs is not None):
            self.list_e_a_r_d.append([self.old_obs,self.old_action,reward,done,self.old_distrib])
        dist = self.Pi(torch.tensor(observation))
        #print(dist)
        m = torch.distributions.categorical.Categorical(dist)
        a = m.sample()
        a = a.item()

        self.old_obs = observation
        self.old_action = a
        self.old_distrib = dist

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            self.optimisation()
            self.compteur = 0
            self.list_e_a_r = []
        
        return a
    
    def optimisation(self):
        all_obs = [li[0] for li in self.list_e_a_r_d]
        all_obs = torch.tensor(all_obs)

        action_prise = [li[1] for li in self.list_e_a_r_d]
        action_prise = torch.tensor(action_prise)

        saveR = [li[2] for li in self.list_e_a_r_d]
        saveR = torch.tensor(saveR)
        
        list_avantage = self.calc_avantage()
        list_avantage = torch.tensor(list_avantage)

        cum_reward = []
        R = 0
        for i in range(len(saveR)-1,-1,-1):
            R = saveR[i] + self.gamma * R
            cum_reward.append(R)
        
        #On obtimise le réseaux V
        cum_reward = torch.tensor(cum_reward)
        pred_V = self.V(all_obs)
        pred_V = pred_V.reshape(-1)

        criterion = torch.nn.MSELoss()
        loss_V = criterion(cum_reward,pred_V)
        loss_V.backward()
        self.optimV.step()

        self.Pi_old = copy.deepcopy(self.Pi)
        pred_teta_k = self.Pi_old(all_obs)
        #On optimise le réseaux Pi pour nbstep
        for _ in range(self.nbstep):
            pred_teta = self.Pi(all_obs)
            #the input given is expected to contain log-probabilities and is not restricted to a 2D Tensor. The targets are given as probabilities (i.e. without taking the logarithm).
            #torch.nn.functional.kl_div(input, target)
            DKL_teta_teta_k = torch.nn.functional.kl_div(torch.log(pred_teta), pred_teta_k.detach())
            #print(DKL_teta_teta_k)
            #Optimiser le réseaux Pi
            pred_Pi_all_obs = self.Pi(all_obs)
            pred_Pi_old_all_obs = self.Pi_old(all_obs).detach()
            Pi_act_value = pred_Pi_all_obs[range(len(pred_Pi_all_obs)),action_prise]
            Pi_old_act_value = pred_Pi_old_all_obs[range(len(pred_Pi_old_all_obs)),action_prise]
            ratios = Pi_act_value/Pi_old_act_value

   
            #On rajoute le moins pour maximiser
            loss_Pi = - ( torch.sum(ratios*list_avantage) + self.beta * DKL_teta_teta_k )
            
            loss_Pi.backward()
           
            self.optimPi.step()
            if(DKL_teta_teta_k >= 1.5*self.sigma):
                self.beta = 2*self.beta
            elif(DKL_teta_teta_k <= self.beta/1.5):
                self.beta = self.beta/2
  

    def calc_avantage(self):
        av = []
        av.append(self.list_e_a_r_d[-1][2])
        for indice in range(len(self.list_e_a_r_d)-2,-1,-1):
            r = self.list_e_a_r_d[indice][2]
            obs = torch.tensor(self.list_e_a_r_d[indice][0])
            if(not(done) and len(av) != 0):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])) - self.V(obs) + self.gamma*self.lambd*av[-1]
            elif(not(done)):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])) - self.V(obs)
            else:
                calc = r - self.V(obs)
            av.append(calc)
        return av





if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Enregistrement de l'Agent
    agent = PPOAgent(tailleDesc=8, nbAction=env.action_space.n, nbEvent=200, nbstep=100,lambd=.1, beta=1,gamma=0.001)

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
