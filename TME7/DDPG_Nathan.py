import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch.nn as nn
from collections import deque
from random import sample
import numpy as np
import torch



class QFunction(nn.Module):

    def __init__(self, size_action_space, size_state_space, layers=[200]):
        super(QFunction, self).__init__()

        sizeEntry = size_action_space + size_state_space
        self.layers = [nn.Linear(sizeEntry,layers[0]), nn.LeakyReLU()]
        for i in range(1,len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[len(layers)-1],1))
        #self.layers = [nn.Linear(sizeEntry,layers[0])] + [nn.Linear(layers[i-1], layers[i]) for i in range(1,len(layers))] + [nn.Linear(layers[len(layers)-1],1)]
        self.monSeq = nn.Sequential(*self.layers)
        print(self.monSeq)

    def forward(self, s, a):

        # A VERIFIER LES FORMATS
        toFeed = torch.cat((s,a),dim=1)
        result = self.monSeq(toFeed)
        return result


class PolicyFunction(nn.Module):
    def __init__(self, size_state_space, size_action_space, layers=[200]):
        super(PolicyFunction, self).__init__()

        self.layers = [nn.Linear(size_state_space,layers[0]), nn.LeakyReLU()]
        for i in range(1,len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[len(layers)-1],size_action_space))
        #self.layers.append(nn.Tanh())
        #self.layers = [nn.Linear(size_state_space, layers[0])] + [nn.Linear(layers[i-1], layers[i]) for i in range(1,len(layers))] + [nn.Linear(layers[len(layers)-1], size_action_space)]
        self.monSeq = nn.Sequential(*self.layers)

        print(self.monSeq)
        

    def forward(self, s):
        result = self.monSeq(s)
        return result

        

class DDPG(object):

    def __init__(self, size_replay_buffer, size_action_space, size_state_space, time_to_update, nb_of_update, batch_size, gamma, ro, a_low, a_high):

        # Replay Buffer
        self.replay_buffer = deque([],maxlen=size_replay_buffer) # Replay buffer

        # Neural nets
        self.Q = QFunction(size_action_space, size_state_space, layers=[200])
        self.Q_target = copy.deepcopy(self.Q)
        self.policy = PolicyFunction(size_state_space, size_action_space, layers=[200])
        self.policy_target = copy.deepcopy(self.policy)

        # Optim
        self.optimQ = torch.optim.Adam(self.Q.parameters())
        self.optimPolicy = torch.optim.Adam(self.policy.parameters())

        #loss
        self.lossMSE = nn.MSELoss()

        # Little variables
        self.batch_size = batch_size
        self.lastA = None
        self.lastState = None
        self.t = 0
        self.time_to_update = time_to_update
        self.nb_of_update = nb_of_update
        self.gamma = gamma
        self.ro = ro
        self.a_low = a_low
        self.a_high = a_high


    def act(self, observation, reward, done):

        # On rend observation sous forme de tensor pour nos neural nets
        observation = torch.tensor(observation, dtype=torch.float)

        # If t = 0 on fait juste une action
        if self.t == 0:
            with torch.no_grad():
                actionToTake = self.policy(observation) + torch.normal(0,1,size=(1,))
                actionToTake[0] = max(self.a_low, min(actionToTake[0], self.a_high))
        

        # Si ce n'est pas le premier pas de temps
        else:

            # On ajoute notre experience au buffer
            self.replay_buffer.append((self.lastState, self.lastA, reward, observation, done))

            # On update nos gradients avec les conditions suivantes
            # Si on a assez un buffer assez grand
            if len(self.replay_buffer) >= self.batch_size:
                
                # Si on est sur un time step où l'on veut update
                if self.t % self.time_to_update == 0:

                    # On update le nombre de fois prévu
                    for z in range(self.nb_of_update):

                        # On récupère un batch
                        chosenIndex = sample([i for i in range(len(self.replay_buffer))], self.batch_size)
                        batch = [self.replay_buffer[i] for i in chosenIndex]

                        # On remplace batch par des tensors utiles pour les neural nets
                        batch = list(zip(*batch)) 
                        batch[0] = torch.stack(batch[0])                    # On stacke les s
                        batch[1] = torch.stack(batch[1])                    # On stack les actions
                        batch[2] = torch.tensor(batch[2]).view((-1,1))      # On stack les rewards
                        batch[3] = torch.stack(batch[3])                    # On stack les sPrimes
                        batch[4] = torch.tensor(batch[4], dtype=torch.float).view((-1,1))  # On stack les done sous forme de int

                        
                        
                        # On calcul les targets y
                        with torch.no_grad():
                            QTargetVal = self.Q_target(batch[3], self.policy_target(batch[3]))
                            y = batch[2] + self.gamma * (1-batch[4]) * QTargetVal

                            
                        # On update la Q function
                        qVal = self.Q(batch[0], batch[1])
                        loss = self.lossMSE(qVal, y)
                        loss.backward()
                        self.optimQ.step()
                        self.optimQ.zero_grad()

                        # On update la policy
                        actVal = self.policy(batch[0])
                        qVal = self.Q(batch[0], actVal)
                        loss = -torch.mean(qVal)
                        loss.backward()
                        #print(self.policy.layers[0].weight.grad)
                        self.optimPolicy.step()
                        self.optimPolicy.zero_grad()
                        

                        # On fait les softs updates des targets
                        for target_param, local_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                            target_param.data.copy_(self.ro*local_param.data + (1.0-self.ro)*target_param.data)

                        for target_param, local_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                            target_param.data.copy_(self.ro*local_param.data + (1.0-self.ro)*target_param.data)
                        




                
            # On calcule notre prochain move:
            with torch.no_grad():
                actionToTake = self.policy(observation) + torch.normal(0,1,size=(1,))
                actionToTake[0] = max(self.a_low, min(actionToTake[0], self.a_high))

                
        # On return ce qu'on a calculé
        self.lastA = actionToTake
        self.lastState = observation
        self.t += 1
        return actionToTake.numpy()

            

if __name__ == '__main__':


    agentN = "DDPG"
    style = "pendulum"
    rTotal = []

    if style == "mountainCar":
        env = gym.make('MountainCarContinuous-v0')
        nbAct = 1
        dimFeatures = 2
        agent =  DDPG(10**6, nbAct, dimFeatures, 50, 20, 50, 0.99, 0.9, -1, 1)
    elif style == "lunar":
        env = gym.make('LunarLanderContinuous-v2')
        nbAct = 2 # A TROUVER
        dimFeatures = 8
        agent =  DDPG(10**6, nbAct, dimFeatures, 50, 200, 60, 0.99, 0.9, -1, 1)
    elif style == "pendulum":
        env = gym.make('Pendulum-v0')
        nbAct = 1 # A TROUVER
        dimFeatures = 3
        agent =  DDPG(10**6, nbAct, dimFeatures, 50, 200, 60, 0.99, 0.9, -1, 1)
    
    #env = gym.make('MountainCarContinuous-v0')

    # Enregistrement de l'Agent
    #agent = DDPG(10**6, 1, 2, 50, 200, 60, 0.99, 0.9, -1, 1)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 200
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        #print("IIIIII : ",i)
        obs = envm.reset()
        env.verbose = (i % 10000 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            #print("obs : ",obs)
            #print("reward : ",reward)
            action = agent.act(obs, reward, done)
            #action = action.numpy()
            #print("action main :",action)
            obs, reward, done, _ = envm.step(action)
            #print("obs : ",obs)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                rTotal.append(rsum)
                break

    print("done")
    env.close()

    nameToGive = agentN + "_" + style
    np.save("../../8/res/"+nameToGive, rTotal)
