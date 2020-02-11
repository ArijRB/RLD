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
from collections import deque
import torch
import torch.nn as nn


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



class PolicyNet(nn.Module):
    def __init__(self, inSize, outSize, layers=[256,128]):
        super(PolicyNet, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        # besoin de tanh ?
        return self.tanh(x)



class QNet(nn.Module):

    def __init__(self, inSize, outSize, layers=[256,128]):
        super(QNet, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.softmax = nn.Softmax()


    def forward(self, state, actionList):
        # A voir au niveau des dimensions etc pour le concatenation
        x = torch.cat((state, actionList), dim=1)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x






class MADDPG:

    def __init__(self, size_buffer, nb_agent, size_observation, size_action, size_batch, gamma, tau, explo=0.05):
        # Les réseaux de neurones (policy target + vrai pour chaque agent, réseau Q pour chaque agent).
        # On utilise un replay buffer, liste de tuple(state, action, reward, state+1)

        # les réseaux
        self.size_input_Q = size_observation + size_action * nb_agent
        self.Q_nets = [QNet(self.size_input_Q,1) for _ in range(nb_agent)]
        self.Q_targets = copy.deepcopy(self.Q_nets)
        self.policy_nets = [PolicyNet(size_observation,size_action) for _ in range(nb_agent)]
        self.policy_targets = copy.deepcopy(self.policy_nets)

        # les optimiseurs
        self.optim_Qs = [torch.optim.Adam(self.Q_nets[i].parameters()) for i in range(nb_agent)]
        self.optim_policies = [torch.optim.Adam(self.policy_nets[i].parameters()) for i in range(nb_agent)]

        # les loss que l'on utilise
        self.criterionMSE = nn.MSELoss()

        # le buffer d'experience replay
        self.buffer_replay = deque([],maxlen=size_buffer)

        # Les variables de sauvegardes des actions/observations
        self.lastObs = None
        self.lastAction = None

        # La distribution Normal d'exploration
        self.normal_explo = torch.distributions.normal.Normal(0,explo)

        # les variables utiles
        self.size_buffer = size_buffer
        self.nb_agent = nb_agent
        self.size_observation = size_observation
        self.size_action = size_action
        self.size_batch = size_batch
        self.gamma = gamma
        self.tau = tau
        self.t = 0


    def act(self, observation, reward, done):
        # ne pas oublier de retransformer ce qu'il faut en torch tensor
        #print("AHAHAHH",reward)

        # Tant qu'on a pas assez de sample pour faire 1.5 batchs, on sauvegarde juste dans le replay_buffer
        if len(self.buffer_replay) <= self.size_batch * 1:
            if self.t > 0:
                self.buffer_replay.append( (self.lastObs, self.lastAction, reward, done, observation) )

            # on prend des actions au pif
            #actionsToTake = []
            #for _ in range(self.nb_agent):
            #    actionsToTake.append((np.random.rand(self.size_action)-0.5)*2)


        # Quand on peut tirer un mini_batch, on commence l'apprentissage
        else:
            # on ajoute les dernières observations:
            #print("BOBOBOBOBO")
            self.buffer_replay.append( (self.lastObs, self.lastAction, reward, done, observation) )

            # Pour chaque agent
            for currentAgent in range(self.nb_agent):

                # on récupère notre batch
                indexToChoose = np.random.choice(len(self.buffer_replay), size=self.size_batch, replace=False)
                batch = [self.buffer_replay[i] for i in indexToChoose]

                # on unzip le batch je pense
                listeStateNow, listeActions, listeReward, _ , listeStateLater = list(zip(*batch))

                # On tensorise toussa
                listeStateNow = torch.tensor(listeStateNow).float()
                listeActions = torch.tensor(listeActions).float()
                listeReward = torch.tensor(listeReward).float()
                listeStateLater = torch.tensor(listeStateLater).float()


                ###### Y_I CALCULUS ##########

                # Maintenant on s'amuse à récupérer les y
                # On doit récupérer les actions pour chaque agent en fonction du stateLater
                actionsPrime = []
                for agent_k in range(self.nb_agent):
                    obsToUse = listeStateLater[range(self.size_batch), np.zeros(self.size_batch) + agent_k, :]
                    with torch.no_grad():
                        actionBatchK = self.policy_targets[agent_k](obsToUse)
                    actionsPrime.append(actionBatchK)
                actionsTensor = torch.cat(actionsPrime,dim=1)

                # on envoie tout ça dans la fonction Q
                with torch.no_grad():
                    Q_prime = self.Q_targets[currentAgent](listeStateLater[range(self.size_batch), np.zeros(self.size_batch) + currentAgent, :], actionsTensor)

                Q_prime = Q_prime.view(-1)
                # On fait le calcul des y_i
                r_i = listeReward[range(self.size_batch),np.zeros(self.size_batch) + currentAgent]
                y_i = r_i + (self.gamma * Q_prime)

                # On fait une descente de gradient sur Q
                # D'abord on récupère les actions
                actionsPrime = []
                for agent_k in range(self.nb_agent):
                    #obsToUse = listeStateNow[range(self.size_batch), np.zeros(self.size_batch) + agent_k, :]
                    #actionBatchK = self.policy_nets[agent_k](obsToUse)
                    actionBatchK = listeActions[range(self.size_batch), np.zeros(self.size_batch) + agent_k, :]
                    actionsPrime.append(actionBatchK)
                actionsTensor = torch.cat(actionsPrime,dim=1)

                #print(listeActions.shape)
                #actionsTensor = listeActions[range(self.size_batch),]

                # Puis on calcule les valeurs de Q
                Q_val = self.Q_nets[currentAgent](listeStateNow[range(self.size_batch), np.zeros(self.size_batch) + currentAgent, :], actionsTensor)

                # Enfin on calcule la loss et on optimise
                loss = self.criterionMSE(Q_val, y_i)
                #print("loss critic ", currentAgent, loss)
                if currentAgent == 0:
                    #print("loss agent0 Q:",loss)
                    pass

                loss.backward()
                #print(self.Q_nets[currentAgent].layers[0].weight.grad)
                self.optim_Qs[currentAgent].step()
                self.optim_Qs[currentAgent].zero_grad()



                ######### OPTIMISATION DES POLICIES #########
                # Pour le moment on va juste optimiser la valeur obtenues pour un Q donné

                # on récupère les actions à filer à Q
                actionsPrime = []
                for agent_k in range(self.nb_agent):
                    obsToUse = listeStateNow[range(self.size_batch), np.zeros(self.size_batch) + agent_k, :]
                    actionBatchK = self.policy_nets[agent_k](obsToUse)
                    actionsPrime.append(actionBatchK)
                actionsTensor = torch.cat(actionsPrime,dim=1)

                Q_val = self.Q_nets[currentAgent](listeStateNow[range(self.size_batch), np.zeros(self.size_batch) + currentAgent, :], actionsTensor)
                loss = -1 * torch.mean(Q_val)
                if currentAgent == 0:
                    #print("loss agent0 Policy:",loss)
                    pass
                #print("loss actor ", currentAgent, loss)
                loss.backward()
                #print(self.policy_nets[currentAgent].layers[0].weight.grad)
                self.optim_policies[currentAgent].step()
                self.optim_policies[currentAgent].zero_grad()






            # On met à jour de façon soft les targets
            for currentPolicy in range(self.nb_agent):
                for target_param, local_param in zip(self.policy_targets[currentPolicy].parameters(), self.policy_nets[currentPolicy].parameters()):
                    target_param.data.copy_(self.tau*target_param.data + (1.0-self.tau)*local_param.data)

            for currentQ in range(self.nb_agent):
                for target_param, local_param in zip(self.Q_targets[currentQ].parameters(), self.Q_nets[currentQ].parameters()):
                    target_param.data.copy_(self.tau*target_param.data + (1.0-self.tau)*local_param.data)

        ###### On décide d'une action #####
        actionsToTake = []
        for i in range(self.nb_agent):
            with torch.no_grad():
                action = torch.clamp(self.policy_nets[i](torch.tensor(observation[i]).float()) + self.normal_explo.sample([2]),-1,1)
            actionsToTake.append(action.numpy())



        self.lastObs = observation
        self.lastAction = actionsToTake
        self.t += 1
        #print(actionsToTake)
        return actionsToTake

    # need to get the yi
    #

if __name__ == '__main__':


    env,scenario,world = make_env('simple_spread')

    # mon agent
    agent = MADDPG(10**6, 3, 14, 2, 1000, 0.99, 0.9)

    o = env.reset()
    d = False
    r = None
    reward = []
    rewardGoing = []
    for currentRun in range(10000):
        # a = []
        # for i, _ in enumerate(env.agents):
        #     #a.append((np.random.rand(2)-0.5)*2)
        #     a.append(np.array([10.0,-1.0]))
        a = agent.act(o,r,d)
        o, r, d, i = env.step(a)
        #print("")
        #print("o",o,len(o),o[0].shape)
        #print(o, r, d, i)

        reward.append(r)
        rewardGoing.append(r)

        if currentRun % 10 == 0:
            print("currentRun rewardGoing :", torch.mean(torch.tensor(rewardGoing)))
            rewardGoing = []

        env.render(mode="none")
    #print("reward :",reward)


    env.close()
    #plt.clf()
