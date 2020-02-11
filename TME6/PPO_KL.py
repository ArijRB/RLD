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


class PPO_KL(object):
    #exemple hyper param : tailleDesc=8, nbEvent=4000, nb_step_Pi=80, nb_step_V=80 ,structure_reseaux=[64] ,gamma=0,99 ,epsilon=0.2
    def __init__(self,tailleDesc, nb_action_dispo, nbEvent, nb_step_Pi ,nb_step_V ,structure_reseaux ,gamma ,sigma):
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
        self.sigma = sigma
        self.beta = 0.2
        self.nb_action_dispo = nb_action_dispo
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
        self.all_beta = []

    def act(self, observation, reward, done):
        if(self.old_obs is not None):
            self.buffer_obs.append(self.old_obs)
            self.buffer_act.append(self.old_action)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)

        dist = self.Pi(torch.tensor(observation).float())
        try:
            m = torch.distributions.categorical.Categorical(dist)
            a = m.sample()
            a = a.item()
        except:
            print("ERREUR NAN")
            print(dist)
            print("FIN ERREUR NAN")
            dist_uniform = torch.ones((self.nb_action_dispo))
            dist_uniform = dist_uniform/self.nb_action_dispo
            m = torch.distributions.categorical.Categorical(dist_uniform)
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
        Pi_teta_old_distrib = self.Pi_old(torch.tensor(self.buffer_obs).float()).detach()
        action_prise = torch.tensor(self.buffer_act)
        #On selectionne uniquement la proba de l'action prise
        Pi_teta_old = Pi_teta_old_distrib[range(len(Pi_teta_old_distrib)),action_prise]
        for step_Pi in range(self.nb_step_Pi):
            Pi_teta_distrib = self.Pi(torch.tensor(self.buffer_obs).float())
            Pi_teta = Pi_teta_distrib[range(len(Pi_teta_distrib)),action_prise]
            #ANCIEN DKL
            DKL_teta_teta_k = torch.nn.functional.kl_div(torch.log(Pi_teta_distrib+1e-9), Pi_teta_old_distrib.detach())
            ratio = Pi_teta/Pi_teta_old
            premier_terme = ratio*advantage


            loss = -premier_terme.mean() + self.beta*DKL_teta_teta_k

            self.optim_Pi.zero_grad()
            loss.backward()
            self.optim_Pi.step()

        #On modifie le beta selon la kl entre
        Pi_teta_kp1_distrib = self.Pi(torch.tensor(self.buffer_obs).float())
        Pi_teta_kp1 = Pi_teta_kp1_distrib[range(len(Pi_teta_kp1_distrib)),action_prise]
        DKL_teta_teta_k_and_kp1 = torch.nn.functional.kl_div(torch.log(Pi_teta_kp1_distrib.detach()+1e-9), Pi_teta_old_distrib.detach())
        if(DKL_teta_teta_k_and_kp1 >= 1.5*self.sigma):
            print("DKL trop grand")
            print(DKL_teta_teta_k_and_kp1)
            print(self.beta)
            self.beta = 2*self.beta
        elif(DKL_teta_teta_k_and_kp1 <= self.sigma/1.5):
            print("DKL trop petit")
            print(DKL_teta_teta_k_and_kp1)
            print(self.beta)
            self.beta = self.beta/2
        self.all_beta.append(self.beta)

        #Optimisation du réseaux V
        criterion = torch.nn.MSELoss()
        for _ in range(self.nb_step_V):
            pred_V = self.V(torch.tensor(self.buffer_obs).float()).reshape(-1)
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
    agent = PPO_KL(tailleDesc=taille_desc,nb_action_dispo=env.action_space.n , nbEvent=2000, nb_step_Pi=80, nb_step_V=80 ,structure_reseaux=[256] ,gamma=0.99 ,sigma=0.1)
    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 2000
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

    windows=30
    plt.plot([np.mean(all_reward[k-windows:k+windows]) for k in range(len(all_reward)-windows)])
    plt.show()

    plt.plot(agent.all_beta)
    plt.show()
