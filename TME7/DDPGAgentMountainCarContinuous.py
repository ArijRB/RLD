import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import gym
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import random
from models import NN2,NN1


class DDPGAgent(object):

    def __init__(self,tailleDesc, low, high, number_of_update=1,batch_size=100 ,freq_update=100, gamma=0.99,tau=0.999,maxlen=100000):
        self.Q= NN2(tailleDesc,1)
        self.Q_target=copy.deepcopy(self.Q)
        self.mu = NN1(tailleDesc,1)
        self.mu_target =copy.deepcopy(self.mu)
        self.freq_update = freq_update
        self.number_of_update = number_of_update
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau=tau 
        self.maxlen = maxlen
        self.compteur = 0
        self.list_o_a_r_d = [()for i in range(self.maxlen)]
        self.old_obs = None
        self.old_action = None
        self.low = low
        self.high = high
        self.start = 0
        self.length = 0
        self.criterion = torch.nn.MSELoss()
        self.optimizer_Q= torch.optim.Adam(self.Q.parameters())
        self.optimizer_mu = torch.optim.Adam(self.mu.parameters())
    


    def act(self, observation, reward, done):
        bruit = np.random.normal()
        observation = torch.tensor(observation).float()
        action_pred = self.mu(observation)[0]+bruit
        action = torch.clamp(action_pred,self.low,self.high).item()
        if self.old_obs is not None:
            if self.length < self.maxlen:
                self.length += 1
            elif self.length == self.maxlen:
                self.start = (self.start + 1) % self.maxlen
            mem = (self.old_obs,self.old_action,reward,observation,done)
            self.list_o_a_r_d[(self.start + self.length - 1) % self.maxlen]=mem
        if(self.compteur > self.freq_update):
            for _ in range(self.number_of_update):
                B = random.choices(self.list_o_a_r_d[:self.length], k=self.batch_size)
                old_obs_batch=[]
                old_action_batch=[]
                reward_batch=[]
                observations_batch=[]
                terminal=[]

                for e in B:
                    print(e)
                    old_obs_batch.append(e[0].numpy())
                    old_action_batch.append(e[1])
                    reward_batch.append(e[2])
                    observations_batch.append(e[3].numpy())
                    if e[4]:
                        terminal.append([1.0])
                    else:
                        terminal.append([0.0])
                old_obs_batch=torch.tensor(np.array(old_obs_batch)).reshape(self.batch_size,-1)
                old_action_batch=torch.tensor(np.array(old_action_batch).reshape(self.batch_size,-1),dtype=torch.float32)
                reward_batch=torch.tensor(np.array(reward_batch).reshape(self.batch_size,-1),dtype=torch.float32)
                observations_batch=torch.from_numpy(np.array(observations_batch).reshape(self.batch_size,-1))
                terminal=torch.tensor(np.array(terminal).reshape(self.batch_size,-1),dtype=torch.float32)
                next_q_values = self.Q_target([observations_batch,self.mu_target(observations_batch)])                
                x=self.gamma*(1.0-terminal)*next_q_values
                target_q_batch = reward_batch+x
                self.Q.zero_grad()
                q_batch = self.Q([old_obs_batch,old_action_batch])        
                loss = self.criterion(q_batch, target_q_batch)
                loss.backward()
                self.optimizer_Q.step()
                self.mu.zero_grad()
                loss_ = -self.Q([observations_batch,self.mu(observations_batch)])
                loss_ =loss_.mean()
                loss_.backward()
                self.optimizer_mu.step()
                self.compteur=0
                # Target update
                
                for target_param, param in zip(self.mu_target.parameters(), self.mu.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)




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

    episode_count = 100000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    env._max_episode_steps = 200
    all_rewards=[]
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
                all_rewards.append(rsum)
                break

    print("done")
    env.close()
    plt.plot(all_rewards)
    plt.show()