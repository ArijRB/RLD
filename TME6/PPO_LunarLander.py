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
import matplotlib.pyplot as plt


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


class PPO_classiqueAgent(object):
    """The world's simplest agent!"""
    def __init__(self,tailleDesc, nbAction, nbEvent, nbstep,lambd, beta=1,gamma=0.9,sigma=1e-3):
        self.V = NN_V(tailleDesc,1,[64])
        self.optimV = torch.optim.Adam(self.V.parameters())
        self.Pi = NN_Pi(tailleDesc,nbAction,[64])
        self.optimPi = torch.optim.Adam(self.Pi.parameters())
        self.Pi_old = NN_Pi(tailleDesc,nbAction,[64])
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
        self.all_beta = []

    def act(self, observation, reward, done):
        observation = observation
        if(self.old_obs is not None):
            self.list_e_a_r_d.append([self.old_obs,self.old_action,reward,done,self.old_distrib])
        dist = self.Pi(torch.tensor(observation).float())
        #print(dist)
        m = torch.distributions.categorical.Categorical(dist)
        a = m.sample()
        a = a.item()

        self.old_obs = observation
        self.old_action = a
        self.old_distrib = dist.detach()

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            self.optimisation()
            self.compteur = 0
            self.list_e_a_r_d = []
        
        return a
    
    def optimisation(self):
        all_obs = [li[0] for li in self.list_e_a_r_d]
        all_obs = torch.tensor(all_obs).float()

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
        loss_V = criterion(cum_reward.detach(),pred_V)
        self.optimV.zero_grad()
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
            #PAS SUR POUR LE DETACH
            pred_Pi_all_obs = self.Pi(all_obs)
            pred_Pi_old_all_obs = self.Pi_old(all_obs).detach()
            Pi_act_value = pred_Pi_all_obs[range(len(pred_Pi_all_obs)),action_prise]
            Pi_old_act_value = pred_Pi_old_all_obs[range(len(pred_Pi_old_all_obs)),action_prise]
            ratios = Pi_act_value/Pi_old_act_value

            self.optimPi.zero_grad()
            #On rajoute le moins pour maximiser
            #PAS SUR DU LOG sur PI
            loss_Pi = - ( torch.sum(torch.log(ratios)*list_avantage) + self.beta * DKL_teta_teta_k )
            
            loss_Pi.backward()
           
            self.optimPi.step()
            if(DKL_teta_teta_k >= 1.5*self.sigma):
                print("DKL trop grand")
                print(DKL_teta_teta_k)
                self.beta = 2*self.beta
            elif(DKL_teta_teta_k <= self.sigma/1.5):
                print("DKL trop petit")
                print(DKL_teta_teta_k)
                self.beta = self.beta/2
            self.all_beta.append(self.beta)
  

    def calc_avantage(self):
        #Version simplifié
        #result = torch.tensor([li[2] for li in self.list_e_a_r_d]).detach().numpy() - self.V(torch.tensor([li[0] for li in self.list_e_a_r_d]).float().detach()).detach().numpy()
        #Version 2 : peut etre nul
        print(len(list_e_a_r_d))
        obs_st = [0]+[li[0] for li in self.list_e_a_r_d[1:]]
        print(len(obs_st))
        result = torch.tensor([li[2] for li in self.list_e_a_r_d]).detach().numpy() - self.gamma*self.V(torch.tensor([li[0] for li in self.list_e_a_r_d]).float().detach()).detach().numpy() - self.V(obs_st).detach()
        return result
        

        #Version trace eligibilité qui ne fonctionne pas pour l'instant
        """
        av = []
        av.append(self.list_e_a_r_d[-1][2])
        for indice in range(len(self.list_e_a_r_d)-2,-1,-1):
            r = self.list_e_a_r_d[indice][2]
            obs = torch.tensor(self.list_e_a_r_d[indice][0])
            done = torch.tensor(self.list_e_a_r_d[indice][3])
            if(not(done) and len(av) != 0):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach() + self.gamma*self.lambd*av[-1]
            elif(not(done)):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach()
            else:
                calc = r - self.V(obs).detach()
            av.append(calc)
        """
    
        return av

class PPO_clipedAgent(object):
    """The world's simplest agent!"""
    def __init__(self,tailleDesc, nbAction, nbEvent, nbstep,lambd ,gamma=0.9,sigma=0.01,epsilon=1e-3):
        self.V = NN_V(tailleDesc,1,[64])
        self.optimV = torch.optim.Adam(self.V.parameters(),lr=0.002)
        self.Pi = NN_Pi(tailleDesc,nbAction,[64])
        self.optimPi = torch.optim.Adam(self.Pi.parameters(),lr=0.002)
        self.Pi_old = NN_Pi(tailleDesc,nbAction,[64])
        self.nbAction = nbAction
        self.nbEvent = nbEvent
        self.sigma = sigma
        self.gamma = gamma
        self.nbstep = nbstep
        self.lambd = lambd
        self.epsilon = epsilon
        self.compteur = 0
        self.list_e_a_r_d = []
        self.old_obs = None
        self.old_action = None
        self.old_distrib = None

    def act(self, observation, reward, done):
        observation = observation
        if(self.old_obs is not None):
            self.list_e_a_r_d.append([self.old_obs,self.old_action,reward,done,self.old_distrib])
        dist = self.Pi(torch.tensor(observation).float())
        #print(dist)
        m = torch.distributions.categorical.Categorical(dist)
        a = m.sample()
        a = a.item()

        self.old_obs = observation
        self.old_action = a
        self.old_distrib = dist.detach()

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            self.optimisation()
            self.compteur = 0
            self.list_e_a_r_d = []
        
        return a
    
    def optimisation(self):
        all_obs = [li[0] for li in self.list_e_a_r_d]
        all_obs = torch.tensor(all_obs).float()
        all_obs = all_obs.detach()

        action_prise = [li[1] for li in self.list_e_a_r_d]
        action_prise = torch.tensor(action_prise)
        action_prise = action_prise.detach()

        saveR = [li[2] for li in self.list_e_a_r_d]
        saveR = torch.tensor(saveR)

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
        loss_V = criterion(cum_reward.detach(),pred_V)
        self.optimV.zero_grad()
        loss_V.backward()
        self.optimV.step()

        self.Pi_old = copy.deepcopy(self.Pi)

        #On optimise le réseaux Pi pour nbstep
        for _ in range(self.nbstep):
            list_avantage = self.calc_avantage()
            list_avantage = torch.tensor(list_avantage)
            #Optimiser le réseaux Pi
            #PAS SUR POUR LES DETACH
            pred_Pi_all_obs = self.Pi(all_obs)
            pred_Pi_old_all_obs = self.Pi_old(all_obs).detach()
            Pi_act_value = pred_Pi_all_obs[range(len(pred_Pi_all_obs)),action_prise]
            Pi_old_act_value = pred_Pi_old_all_obs[range(len(pred_Pi_old_all_obs)),action_prise]

            ratios = Pi_act_value/Pi_old_act_value.detach()

            
            #On rajoute le moins pour maximiser
            #PAS SUR DU LOG sur PI
            premier_terme_min = ratios*list_avantage
            deuxieme_terme_min = torch.clamp(ratios,1-self.epsilon,1+self.epsilon)*list_avantage
            #print("###############")
            #print(ratios[0])
            #print(1-self.epsilon)
            #print(1+self.epsilon)
            #print(premier_terme_min.shape)
            #print(deuxieme_terme_min.shape)
            #print(premier_terme_min[0])
            #print(deuxieme_terme_min[0])
            mini = torch.min(premier_terme_min,deuxieme_terme_min)
            #print(mini[0])
            #print(mini.shape)
            loss_Pi = -torch.sum(mini)
            #print(loss_Pi.shape)
            #print("###############")
            self.optimPi.zero_grad()
            loss_Pi.backward()
            self.optimPi.step()
  

    def calc_avantage(self):
        #Version simplifié
        #result = torch.tensor([li[2] for li in self.list_e_a_r_d]).detach().numpy() - self.V(torch.tensor([li[0] for li in self.list_e_a_r_d]).float().detach()).detach().numpy()
        #Version 2 : peut etre faux
        #print(len(self.list_e_a_r_d))
        obs_st = torch.tensor([li[0] for li in self.list_e_a_r_d[1:]]).float().detach()
        #print(obs_st[0])
        #print(torch.tensor(self.list_e_a_r_d[0][0]).float().shape)
        obs_st = torch.cat((torch.tensor(self.list_e_a_r_d[0][0]).float().reshape(1,-1),obs_st))
        #print(obs_st.shape)
        result = torch.tensor([li[2] for li in self.list_e_a_r_d]).detach().numpy() - self.gamma*self.V(torch.tensor([li[0] for li in self.list_e_a_r_d]).float().detach()).detach().numpy() - self.V(obs_st).float().detach().numpy()
        return result
        

        #Version trace eligibilité qui ne fonctionne pas pour l'instant
        """
        av = []
        av.append(self.list_e_a_r_d[-1][2])
        for indice in range(len(self.list_e_a_r_d)-2,-1,-1):
            r = self.list_e_a_r_d[indice][2]
            obs = torch.tensor(self.list_e_a_r_d[indice][0])
            done = torch.tensor(self.list_e_a_r_d[indice][3])
            if(not(done) and len(av) != 0):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach() + self.gamma*self.lambd*av[-1]
            elif(not(done)):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach()
            else:
                calc = r - self.V(obs).detach()
            av.append(calc)
        """
    
        return av


class PPO_clipedAgent_v42(object):
    #exemple hyper param : tailleDesc=8, nbEvent=4000, nb_step_Pi=80, nb_step_V=80 ,structure_reseaux=[64] ,gamma=0,99 ,epsilon=0.2
    def __init__(self,tailleDesc, nb_action_dispo, nbEvent, nb_step_Pi ,nb_step_V ,structure_reseaux ,gamma ,epsilon):
        #Réseaux et optimiser
        self.V = NN_V(tailleDesc,1,structure_reseaux)
        self.Pi = NN_Pi(tailleDesc,nb_action_dispo,structure_reseaux)
        self.Pi_old = None
        self.optim_V = torch.optim.Adam(self.V.parameters())
        self.optim_Pi = torch.optim.Adam(self.Pi.parameters())
        #hyper paramettre
        self.nbEvent = nbEvent
        self.gamma = gamma
        self.nb_step_Pi = nb_step_Pi
        self.nb_step_V = nb_step_V
        self.epsilon = epsilon
        #Compteur pour savoir quand on a fait nbEvent action et mettre a jours le réseaux
        self.compteur = 0
        #save precedent obs and action
        self.old_obs = None
        self.old_action = None
        #buffer
        self.buffer_obs= []
        self.buffer_act = []
        self.buffer_reward = []
        self.buffer_done = []

    def act(self, observation, reward, done):
        if(self.old_obs is not None):
            self.buffer_obs.append(self.old_obs)
            self.buffer_act.append(self.old_action)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        
        dist = self.Pi(torch.tensor(observation).float())
        m = torch.distributions.categorical.Categorical(dist)
        a = m.sample()
        a = a.item()

        self.old_obs = observation
        self.old_action = a

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            self.optimisation()
            #Renitialisation des buffers et du compteur
            self.compteur = 0
            self.buffer_obs= []
            self.buffer_act = []
            self.buffer_reward = []
            self.buffer_done = []
        return a
    
    def optimisation(self):
        reward_to_go = self.calc_reward_to_go()
        advantage = self.calc_avantage()

        #Optimisation du réseaux Pi
        self.Pi_old = copy.deepcopy(self.Pi)
        #Pi_teta_old est appelé Pi_teta_k dans les slides de cours
        Pi_teta_old = self.Pi_old(torch.tensor(self.buffer_obs).float()).detach()
        action_prise = torch.tensor(self.buffer_act)
        #On selectionne uniquement la proba de l'action prise
        Pi_teta_old = Pi_teta_old[range(len(Pi_teta_old)),action_prise]
        for step_Pi in range(self.nb_step_Pi):
            Pi_teta = self.Pi(torch.tensor(self.buffer_obs).float())
            Pi_teta = Pi_teta[range(len(Pi_teta)),action_prise]
            ratio = Pi_teta/Pi_teta_old
            premier_terme = ratio*advantage
            deuxieme_terme = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*advantage
            loss = -torch.min(premier_terme,deuxieme_terme)
            self.optim_Pi.zero_grad()
            loss.mean().backward()
            self.optim_Pi.step()

        #Optimisation du réseaux V
        criterion = torch.nn.MSELoss()
        for _ in range(self.nb_step_V): 
            pred_V = self.V(torch.tensor(self.buffer_obs).float()).reshape(-1)
            #loss_V = criterion(pred_V,reward_to_go)
            #test loss_V
            loss_V = torch.sum(torch.pow(pred_V-reward_to_go,2))/self.nbEvent
            self.optim_V.zero_grad()
            loss_V.backward()
            self.optim_V.step()

    
    def calc_reward_to_go(self):
        reward_to_go = []
        for k in range(len(self.buffer_reward)-1,-1,-1):
            reward = self.buffer_reward[k]
            done = self.buffer_done[k]
            if(len(reward_to_go) != 0 and not done):
                item_RTG = reward+self.gamma*reward_to_go[-1]
            else:
                item_RTG=reward
            reward_to_go.append(item_RTG)
        #On doit ré inverser la liste pour etre dans le bon ordre
        reward_to_go = list(reversed(reward_to_go))
        return torch.tensor(reward_to_go)


    def calc_avantage(self):
        avantages = []
        for k in range(len(self.buffer_obs)):
            S_t = torch.tensor(self.buffer_obs[k]).float()
            if(self.buffer_done[k] or k==len(self.buffer_obs)-1):
                av = -self.V(S_t)+self.buffer_reward[k]
            else:
                S_tp1 = torch.tensor(self.buffer_obs[k+1]).float()
                av = -self.V(S_t)+self.buffer_reward[k]+self.gamma*self.V(S_tp1)
            avantages.append(av)
        return torch.tensor(avantages).detach()




class PPO_wihtout_all(object):
    """The world's simplest agent!"""
    def __init__(self,tailleDesc, nbAction, nbEvent, nbstep,lambd ,gamma=0.9,sigma=0.01,epsilon=1e-3):
        self.V = NN_V(tailleDesc,1,[64])
        self.optimV = torch.optim.Adam(self.V.parameters(),lr=0.002)
        self.Pi = NN_Pi(tailleDesc,nbAction,[64])
        self.optimPi = torch.optim.Adam(self.Pi.parameters(),lr=0.002)
        self.Pi_old = NN_Pi(tailleDesc,nbAction,[64])
        self.nbAction = nbAction
        self.nbEvent = nbEvent
        self.sigma = sigma
        self.gamma = gamma
        self.nbstep = nbstep
        self.lambd = lambd
        self.epsilon = epsilon
        self.compteur = 0
        self.list_e_a_r_d = []
        self.old_obs = None
        self.old_action = None
        self.old_distrib = None

    def act(self, observation, reward, done):
        observation = observation
        if(self.old_obs is not None):
            self.list_e_a_r_d.append([self.old_obs,self.old_action,reward,done,self.old_distrib])
        dist = self.Pi(torch.tensor(observation).float())
        #print(dist)
        m = torch.distributions.categorical.Categorical(dist)
        a = m.sample()
        a = a.item()

        self.old_obs = observation
        self.old_action = a
        self.old_distrib = dist#.detach()

        self.compteur+=1
        if(self.compteur>self.nbEvent):
            self.optimisation()
            self.compteur = 0
            self.list_e_a_r_d = []
        
        return a
    
    def optimisation(self):
        all_obs = [li[0] for li in self.list_e_a_r_d]
        all_obs = torch.tensor(all_obs).float()
        all_obs = all_obs.detach()

        action_prise = [li[1] for li in self.list_e_a_r_d]
        action_prise = torch.tensor(action_prise)
        action_prise = action_prise#.detach()

        saveR = [li[2] for li in self.list_e_a_r_d]
        saveR = torch.tensor(saveR)

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
        loss_V = criterion(cum_reward.detach(),pred_V)
        self.optimV.zero_grad()
        loss_V.backward()
        self.optimV.step()

        self.Pi_old = copy.deepcopy(self.Pi)

        #On optimise le réseaux Pi pour nbstep
        for _ in range(self.nbstep):
            list_avantage = self.calc_avantage()
            list_avantage = torch.tensor(list_avantage)
            #Optimiser le réseaux Pi
            #PAS SUR POUR LES DETACH
            pred_Pi_all_obs = self.Pi(all_obs)
            pred_Pi_old_all_obs = self.Pi_old(all_obs).detach()
            Pi_act_value = pred_Pi_all_obs[range(len(pred_Pi_all_obs)),action_prise]
            Pi_old_act_value = pred_Pi_old_all_obs[range(len(pred_Pi_old_all_obs)),action_prise]

            ratios = Pi_act_value/Pi_old_act_value.detach()

            
            #On rajoute le moins pour maximiser
            #PAS SUR DU LOG sur PI
            premier_terme_min = ratios*list_avantage
            #deuxieme_terme_min = torch.clamp(ratios,1-self.epsilon,1+self.epsilon)*list_avantage
            #print("###############")
            #print(ratios[0])
            #print(1-self.epsilon)
            #print(1+self.epsilon)
            #print(premier_terme_min.shape)
            #print(deuxieme_terme_min.shape)
            #print(premier_terme_min[0])
            #print(deuxieme_terme_min[0])
            #mini = torch.min(premier_terme_min,deuxieme_terme_min)
            #print(mini[0])
            #print(mini.shape)
            loss_Pi = -torch.sum(premier_terme_min)
            #print(loss_Pi.shape)
            #print("###############")
            self.optimPi.zero_grad()
            loss_Pi.backward()
            self.optimPi.step()
  

    def calc_avantage(self):
        #Version simplifié
        result = torch.tensor([li[2] for li in self.list_e_a_r_d]).detach().numpy() - self.V(torch.tensor([li[0] for li in self.list_e_a_r_d]).float().detach()).detach().numpy()
        return result
        

        #Version trace eligibilité qui ne fonctionne pas pour l'instant
        """
        av = []
        av.append(self.list_e_a_r_d[-1][2])
        for indice in range(len(self.list_e_a_r_d)-2,-1,-1):
            r = self.list_e_a_r_d[indice][2]
            obs = torch.tensor(self.list_e_a_r_d[indice][0])
            done = torch.tensor(self.list_e_a_r_d[indice][3])
            if(not(done) and len(av) != 0):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach() + self.gamma*self.lambd*av[-1]
            elif(not(done)):
                calc = r+self.gamma * self.V(torch.tensor(self.list_e_a_r_d[indice+1][0])).detach() - self.V(obs).detach()
            else:
                calc = r - self.V(obs).detach()
            av.append(calc)
        """
    
        return av



if __name__ == '__main__':
    #num_env : 1 : Cartpole et 2 : lunarlander
    num_env = 2

    if(num_env == 1):
        #Cartpole env
        env = gym.make('CartPole-v1')
        taille_desc = 4
    elif(num_env == 2):
        #Lunar Lander env
        env = gym.make('LunarLander-v2')
        taille_desc = 8
    else:
        print("num error")
        exit(0)

    # Enregistrement de l'Agent
    #Agent PPO classique
    #agent = PPO_classiqueAgent(tailleDesc=taille_desc, nbAction=env.action_space.n, nbEvent=2000, nbstep=10,lambd=.1, beta=1,gamma=0.99)
    #Agent PPO cliped
    #agent = PPO_clipedAgent(tailleDesc=taille_desc, nbAction=env.action_space.n, nbEvent=2000, nbstep=30,lambd=.1,gamma=0.99, epsilon=2e-1)
    #Agent PPO sans clip et sans dkl
    #agent = PPO_wihtout_all(tailleDesc=taille_desc, nbAction=env.action_space.n, nbEvent=2000, nbstep=10,lambd=.1,gamma=0.99, epsilon=2e-1)

    agent = PPO_clipedAgent_v42(tailleDesc=taille_desc,nb_action_dispo=env.action_space.n , nbEvent=2000, nb_step_Pi=80, nb_step_V=80 ,structure_reseaux=[256] ,gamma=0.99 ,epsilon=0.2)
    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 5000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    all_reward = []
    rsum = 0
    env._max_episode_steps = 200
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
                all_reward.append(rsum)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    plt.clf()
    plt.plot(all_reward)
    plt.show()

    plt.plot(agent.all_beta)
    plt.show()
