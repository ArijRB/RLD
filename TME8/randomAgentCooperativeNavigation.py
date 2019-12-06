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
from random import sample 


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
    def __init__(self,number_of_agent,taille_obs,taille_action_space,number_sample,GAMMA=0.99):
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
        for _ in range(number_of_agent):
            Q = NN( (taille_obs+taille_action_space)*number_of_agent,1,layers=[25])
            self.liste_Q.append(Q)
            opti_Q = torch.optim.Adam(self.Q.parameters())
            self.liste_Q_opti.append(opti_Q)
            mu = NN_softmax(taille_obs,taille_action_space)
            self.liste_mu.append(mu)
            mu_target = NN_softmax(taille_obs,taille_action_space)
            self.liste_mu_target.append(mu_target)
        self.experience = []
        self.ancien_obs = None
        self.ancien_action = None
    
    def act(self, obs, reward):
        if(self.ancien_obs != None):
            self.experience.append((self.ancien_obs,self.ancien_action,r,o))
        all_action = []
        for num_agent in range(self.number_of_agent):
            N = np.random.standard_normal(self.taille_action_space)
            obs_agent = torch.tensor(obs[num_agent]).float()
            mu = self.liste_mu[num_agent](obs_agent)
            action = mu.detach().numpy() + N
            print(action)
            all_action.append(np.array(action))
        self.ancien_obs = o
        self.ancien_action = all_action
        return all_action

    def optim_reseaux(self):
        sample = sample(self.experience,self.number_sample)
        for num_agent in range(self.number_of_agent):
            label_agent = []
            batch_pred = torch.tensor([])
            for x, a, r, x_prime in zip(sample):
                #Calcul des labels y_i
                action_with_target_net = torch.tensor([])
                for num_agent2 in range(self.number_of_agent):
                    N = np.random.standard_normal(self.taille_action_space)
                    obs_agent = torch.tensor(x_prime[num_agent2]).float()
                    mu_prime = self.liste_mu_target[num_agent2](obs_agent)
                    action_target = mu + torch.tensor(N)
                    action_with_target_net.cat(action_target)
                entree_Q = torch.cat(x_prime,action_with_target_net)
                Q_value = self.liste_Q[num_agent](entree_Q)
                y_i = r + self.gamma*Q_value
                label_agent.append(y_i)
                #Calcul des predictions du réseaux
                entree_Q = torch.cat(x,a)
                batch_pred = torch.stack(batch_pred,entree_Q)
            self.liste_Q_opti[num_agent].zero_grad()
            pred = self.liste_Q[num_agent](batch_pred)
            loss = self.criterion(pred,label_agent)
            self.liste_Q_opti[num_agent].step()

                




        


if __name__ == '__main__':


    env,scenario,world = make_env('simple_spread')
    o = env.reset()
    decideur = all_agent(len(env.agents),len(o[0]),env.world.dim_p,20)
    r=0
    reward = []
    for _ in range(100):
        print(o)
        a = decideur.act(o,r)
        o, r, d, i = env.step(a)
        print(a)
        reward.append(r)
        env.render(mode="none")
    print(reward)


    env.close()