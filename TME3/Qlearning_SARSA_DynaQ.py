import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt
import time


class QAgent(object):
    def __init__(self, action_space,env):
        self.action_space = action_space
        self.Q = dict()
        self.gamma = 0.99
        self.seuil_greedy = 1
        self.alpha = 0.5
        self.alpha_decay=.9995
        self.eps_decay = 0.999
        self.last_obs = None
        self.last_action = None
        self.env = env

    def act(self, observation, reward, done):
        observation = self.env.state2str(observation)
        if(not(self.last_obs is None)):
            if(self.last_obs not in list(self.Q.keys())):
                self.Q[self.last_obs] = np.zeros(self.action_space.n)
            if(done):
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward -self.Q[self.last_obs][self.last_action])
            else:
                try :
                    m = np.max(self.Q[observation])
                except:
                    m = 0
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward + self.gamma*m -self.Q[self.last_obs][self.last_action])

        if(observation in list(self.Q.keys())):
            if(np.random.random()>self.seuil_greedy):
                maxi = np.max(self.Q[observation])
                action = np.random.choice(np.where(self.Q[observation] == maxi)[0],1)
            else:
                action = np.random.randint(self.action_space.n)
        else:
            action = np.random.randint(self.action_space.n)
        self.seuil_greedy *= self.eps_decay
        self.alpha *= self.alpha_decay

        self.last_obs = observation
        self.last_action = action

        return action

class Sarsa(object):
    def __init__(self, action_space,env):
        self.action_space = action_space
        self.Q = dict()
        self.gamma = 0.99
        self.seuil_greedy = 1
        self.alpha = 0.5
        self.alpha_decay=.9995
        self.eps_decay = 0.999
        self.last_obs = None
        self.last_action = None
        self.env = env

    def act(self, observation, reward, done):
        observation = self.env.state2str(observation)

        if(observation in list(self.Q.keys())):
            if(np.random.random()>self.seuil_greedy):
                maxi = np.max(self.Q[observation])
                action = np.random.choice(np.where(self.Q[observation] == maxi)[0],1)
            else:
                action = np.random.randint(self.action_space.n)
        else:
            action = np.random.randint(self.action_space.n)


        if(not(self.last_obs is None)):
            if(self.last_obs not in list(self.Q.keys())):
                self.Q[self.last_obs] = np.zeros(self.action_space.n)
            if(done):
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward -self.Q[self.last_obs][self.last_action])
            else:
                try :
                    m = self.Q[observation][action]
                except:
                    m=0
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward + self.gamma*m -self.Q[self.last_obs][self.last_action])

        self.seuil_greedy *= self.eps_decay
        self.alpha *= self.alpha_decay
        self.last_obs = observation
        self.last_action = action

        return action

class DynaQ(object):
    def __init__(self, action_space,env):
        self.action_space = action_space
        self.Q = dict()
        self.P = dict()
        self.R = dict()
        self.alpha_r = 0.5
        self.alpha_P = 0.5
        self.k=5
        self.gamma = 0.99
        self.seuil_greedy = 1
        self.alpha = 0.5
        self.alpha_decay=.9995
        self.eps_decay = 0.999
        self.last_obs = None
        self.last_action = None
        self.env = env

    def act(self, observation, reward, done):
        observation = self.env.state2str(observation)
        if(observation not in self.Q):
            self.Q[observation] = np.zeros(self.action_space.n)

        #Update Q
        if(not(self.last_obs is None)):
            if(self.last_obs not in list(self.Q.keys())):
                self.Q[self.last_obs] = np.zeros(self.action_space.n)
            if(done):
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward -self.Q[self.last_obs][self.last_action])
            else:
                try :
                    m = self.Q[observation][action]
                except:
                    m=0
                self.Q[self.last_obs][self.last_action] += self.alpha*(reward + self.gamma*m -self.Q[self.last_obs][self.last_action])

            #update MDP
            try :
                self.R[(self.last_obs,str(self.last_action),observation)] += self.alpha_r*(reward - self.R[(self.last_obs,str(self.last_action),observation)])
            except:
                self.R[(self.last_obs,str(self.last_action),observation)] = 0
            try :
                self.P[(observation,self.last_obs,str(self.last_action))] += self.alpha_P*(1-self.P[(observation,self.last_obs,str(self.last_action))])
            except:
                self.P[(observation,self.last_obs,str(self.last_action))] = 0

            for obs in self.Q.keys() :
                if(obs != observation):
                    try:
                        self.P[(obs,self.last_obs,str(self.last_action))] += self.alpha_P*(1-self.P[(obs,self.last_obs,str(self.last_action))])
                    except:
                        pass


        #Sampling
        for _ in range(self.k):
            obs = np.random.choice(list(self.Q.keys()))
            act = np.random.choice(list(range(self.action_space.n)))
            sum = 0
            for s_prime in self.Q.keys() :
                m = np.max(self.Q[s_prime])
                try:
                    sum += self.P[(s_prime,obs,act)]*(self.R[(obs,act,s_prime)] + self.gamma*m)
                except:
                    pass
            self.Q[obs][act] += self.alpha * (sum - self.Q[obs][act] )

        #Choix de l'action
        if(observation in list(self.Q.keys())):
            if(np.random.random()>self.seuil_greedy):
                maxi = np.max(self.Q[observation])
                action = np.random.choice(np.where(self.Q[observation] == maxi)[0],1)
            else:
                action = np.random.randint(self.action_space.n)
        else:
            action = np.random.randint(self.action_space.n)


        self.seuil_greedy *= self.eps_decay
        self.alpha *= self.alpha_decay
        self.last_obs = observation
        self.last_action = action

        return action

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    #env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console


    # Execution avec un Agent
    agent1 = QAgent(env.action_space,env)
    agent2 = Sarsa(env.action_space,env)
    agent3 = DynaQ(env.action_space,env)
    agents=[(agent1,"QAgent","red"),(agent2,"Sarsa","blue"),(agent3,"DynaQ","green")]

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0:-0.001,3:1,4:1,5:-1,6:-1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    res=[]
    for agent,name,color in agents:
        reward = 0
        done = False
        rsum = 0
        FPS = 0.0001
        all_reward = []
        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
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
                    all_reward.append(rsum)
                    break
            
        #mean_all_reward = []
        #for k in range(len(all_reward)-100):
            #mean_all_reward.append(np.mean(all_reward[k:k+100]))
        res.append((name,color,np.cumsum(all_reward)))
    
    
    plt.clf()
  
    plt.figure()

    plt.title("Cumulated reward")
    for name, color,cum_reward in res:
        plt.plot(cum_reward, label=name, color=color)
    plt.legend()
    plt.show()

    print("done")
    env.close()
