import matplotlib

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
import torch.nn as nn
import numpy as np
import torch
import random
import time
import matplotlib.pyplot as plt


"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

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

class NN_softmax(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN_softmax, self).__init__()
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

class all_agent():
    def __init__(self,number_of_agent,taille_obs,taille_action_space,number_sample,GAMMA=0.99,update_target=5):
        print("Taille obs space",taille_obs)
        print("Action Space size",taille_action_space)
        self.number_of_agent = number_of_agent
        self.taille_obs = taille_obs
        self.taille_action_space = taille_action_space
        self.number_sample = number_sample
        self.gamma = GAMMA
        self.liste_Q = []
        self.liste_mu = []
        self.liste_mu_target = []
        self.liste_Q_opti = []
        self.liste_mu_opti = []
        for _ in range(number_of_agent):
            Q = NN( (taille_obs+taille_action_space)*number_of_agent,1,layers=[256,128])
            self.liste_Q.append(Q)
            opti_Q = torch.optim.Adam(Q.parameters())
            self.liste_Q_opti.append(opti_Q)
            mu = NN_softmax(taille_obs,taille_action_space,layers=[256,128])
            self.liste_mu.append(mu)
            opti_mu = torch.optim.Adam(mu.parameters())
            self.liste_mu_opti.append(opti_mu)
            mu_target = NN_softmax(taille_obs,taille_action_space,layers=[256,128])
            self.liste_mu_target.append(mu_target)
        self.experience = []
        self.ancien_obs = None
        self.ancien_action = None
        self.criterion = torch.nn.MSELoss()
        self.compteur_optim_step=0
        self.update_target = update_target
        self.tau = 0.9

    def act(self, obs, reward):
        if(self.ancien_obs != None):
            self.experience.append((self.ancien_obs,self.ancien_action,r,obs))
        all_action = []
        for num_agent in range(self.number_of_agent):
            N = np.random.standard_normal(self.taille_action_space)
            obs_agent = torch.tensor(obs[num_agent]).float()
            mu = self.liste_mu_target[num_agent](obs_agent)
            action = mu.detach().numpy() + N
            #print(action)
            all_action.append(np.array(action))
        self.ancien_obs = obs
        self.ancien_action = all_action
        if(len(self.experience) > self.number_sample):
            #print("Optimisation du réseaux")
            self.optim_reseaux()
        return all_action

    def optim_reseaux(self):
        #print(len(self.experience))
        sample = random.sample(self.experience,self.number_sample)
        #Mise à jour des valeurs Q
        for num_agent in range(self.number_of_agent):
            label_agent = []
            list_valeur_mu = torch.tensor([])
            batch_pred = torch.tensor([])
            for x, a, r, x_prime in sample:
                #Calcul des labels y_i
                action_with_target_net = torch.tensor([])
                for num_agent2 in range(self.number_of_agent):
                    N = np.random.standard_normal(self.taille_action_space)
                    obs_agent = torch.tensor(x_prime[num_agent2]).float()
                    mu_prime = self.liste_mu_target[num_agent2](obs_agent).detach()
                    action_target = mu_prime + torch.tensor(N)
                    action_with_target_net = torch.cat((action_with_target_net,action_target.float()))
                entree_Q = torch.cat((torch.tensor(x_prime).reshape(-1).float(),action_with_target_net))
                Q_value = self.liste_Q[num_agent](entree_Q).detach()
                #print(num_agent)
                #print("longueur de r",len(r))
                y_i = torch.tensor(r[num_agent]) + self.gamma*Q_value

                label_agent.append(y_i)#.detach().numpy())
                #Calcul des predictions du réseaux
                entree_Q = torch.cat((torch.tensor(x).reshape(-1),torch.tensor(a).reshape(-1)))
                batch_pred = torch.cat((batch_pred.float(),entree_Q.reshape(1,-1).float()))
                #Calcul pour la mise à jour de la politique
                mu = self.liste_mu[num_agent](torch.tensor(x[num_agent]).float())
                list_valeur_mu = torch.cat((list_valeur_mu.float(),mu.reshape(1,-1).float()))

            pred = self.liste_Q[num_agent](batch_pred)
            pred = pred.reshape(-1)
            #pred c'est Q_i
            #Mise a jours de la politique
            #ATTENTION DERIVE UNIQUEMENT PAR RAPPORT A a_i DONC FAIRE DETACH POUR LE RESTE
            #Somme
            actor_maximise = torch.sum(torch.t(list_valeur_mu)*pred.detach())
            actor_maximise = actor_maximise/self.number_sample
            actor_minimise = -actor_maximise
            #print("On veut minimiser -actor_maximise :",actor_maximise.item())
            self.liste_mu_opti[num_agent].zero_grad()
            actor_minimise.backward()#retain_graph=True)
            self.liste_mu_opti[num_agent].step()

            pred = self.liste_Q[num_agent](batch_pred)
            pred = pred.reshape(-1)

            label_agent = torch.tensor(label_agent).reshape(-1)
            loss = self.criterion(pred,label_agent)
            #print("On veut minimiser MSE Q :",loss.item())
            self.liste_Q_opti[num_agent].zero_grad()
            loss.backward()
            self.liste_Q_opti[num_agent].step()
        self.compteur_optim_step+=1

        """
        if(self.compteur_optim_step > self.update_target):
            #print("Je transfert le target")
            self.liste_mu_target = copy.deepcopy(self.liste_mu)
            self.compteur_optim_step=0
        """
        #ATTENTION peut etre inversé le target et le normal
        # On met à jour de façon soft les targets
        for currentPolicy in range(self.number_of_agent):
            for target_param, local_param in zip(self.liste_mu_target[currentPolicy].parameters(), self.liste_mu[currentPolicy].parameters()):
                target_param.data.copy_(self.tau*target_param.data + (1.0-self.tau)*local_param.data)




if __name__ == '__main__':
    env,scenario,world = make_env('simple_spread')
    all_multiple_reward = []
    rewardGoing = []
    decideur = all_agent(number_of_agent=len(env.agents),taille_obs=14,taille_action_space=2,number_sample=400)
    for _ in range(100):
        o = env.reset()
        r=torch.zeros((len(env.agents)))
        reward = []
        for currentRun in range(1000):
            #print(o)
            a = decideur.act(o,r)
            o, r, d, i = env.step(a)
            #print(a)
            rewardGoing.append(r)
            reward.append(r)
            env.render(mode="none")
            if currentRun % 10 == 0:
                print(currentRun)
                print("currentRun rewardGoing :", torch.mean(torch.tensor(rewardGoing)))
                rewardGoing = []

        print(np.sum(reward))
        all_multiple_reward.append(np.sum(reward))

    env.close()
    plt.plot(all_multiple_reward)
    plt.show()
