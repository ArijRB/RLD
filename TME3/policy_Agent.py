import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt


class QAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.Q = dict()
        self.gamma = 0.9
        self.seuil_gready = 0.95
        self.alpha = 1e-2
        self.last_obs = None
        self.last_action = None

    def act(self, observation, reward, done):
        if(not(self.last_obs is None)):
            if(str(self.last_obs) not in list(self.Q.keys())):
                self.Q[str(self.last_obs)] = np.zeros(self.action_space.n)
            if(done):
                self.Q[str(self.last_obs)][self.last_action] = self.Q[str(self.last_obs)][self.last_action] + self.alpha*(reward -self.Q[str(self.last_obs)][self.last_action])
            else:
                self.Q[str(self.last_obs)][self.last_action] = self.Q[str(self.last_obs)][self.last_action] + self.alpha*(reward + self.gamma*np.max(self.Q[str(observation)])-self.Q[str(self.last_obs)][self.last_action])
        

        if(str(observation) in list(self.Q.keys())):
            if(np.random.random()>self.seuil_gready):
                maxi = np.max(self.Q[str(observation)])
                action = np.random.choice(np.where(self.Q[str(observation)] == maxi)[0],1)
            else:
                action = np.random.randint(self.action_space.n)
        else:
            action = np.random.randint(self.action_space.n)
        

        self.last_obs = observation
        self.last_action = action

        return action


if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console

   
    # Execution avec un Agent
    agent = QAgent(env.action_space)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0:-0.001,3:1,4:1,5:-1,6:-1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    all_reward = []
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
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
                break
        all_reward.append(rsum)
    plt.clf()
    plt.plot(all_reward)
    plt.show()  

    print("done")
    env.close()
