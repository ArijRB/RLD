import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

#Policy iteration keyvan
class PolicyIterationAgentV2:
    def __init__(self, state_space, action_space, p, gamma=.99):
        self.policy = None
        self.state_values = None
        self.action_space = action_space
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.last_state = None
    
    def get_state_values_from_policy(self, policy, eps=1e-6):
        current_state_values = self.state_values.copy()
        new_state_values = {}
        nb_etape_v = 0
        tableau_diff = []
        diff = eps
        while diff >= eps:
            
            for state in self.state_space:
                state_value = 0
                
                for proba, state2, reward, done in self.p[state][policy[state]]:
                    val = reward
                    if not done:
                        val += self.gamma * current_state_values[state2]
                    val *= proba
                    
                    state_value += val
                
                new_state_values[state] = state_value
                assert current_state_values is not new_state_values
            
            diff = 0
            for state in self.state_space:
                diff += (new_state_values[state] - current_state_values[state]) ** 2
                current_state_values[state] = new_state_values[state]
            diff = np.sqrt(diff)
            tableau_diff.append(diff)
            nb_etape_v+=1
        
        return new_state_values,nb_etape_v,tableau_diff
    
    def _update_policy(self):
        new_policy = {}
        for state in self.state_space:
            best_action = None
            max_value = -np.inf
            
            for action in self.p[state].keys():
                s = 0
                for proba, state2, reward, done in self.p[state][action]:
                    val = reward
                    if not done:
                        val += self.gamma * self.state_values[state2]
                    val *= proba
                    s += val
                
                if s > max_value:
                    best_action = action
                    max_value = s
            
            new_policy[state] = best_action
        
        return new_policy
    
    def compute_best_policy(self, eps=1e-6):
        # initialisation de la politique aléatoirement
        self.policy = {state: self.action_space.sample() for state in self.state_space}
        
        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in self.state_space}
        nb_etape_totale = 0
        nb_total_v = 0
        all_diff = []
        change = True
        while change:
            # on ne réinitialise pas V à chaque itération
            self.state_values,nb_etape_v,tableau_diff = self.get_state_values_from_policy(self.policy, eps)
            all_diff = all_diff + tableau_diff
            nb_total_v+= nb_etape_v
            new_policy = self._update_policy()
            
            change = False
            for state in self.policy.keys():
                if self.policy[state] != new_policy[state]:
                    change = True
                    break
            nb_etape_totale+=1
            self.policy = new_policy
        return nb_etape_totale,nb_total_v,all_diff
    
    def act(self):
        return self.policy[self.last_state]
    
    def reset(self, obs):
        self.last_state = obs
    
    def get_result(self, obs, _reward, _done):
        self.last_state = obs
    
    def evaluate_other_agent_policy(self, agent, eps=1e-3):
        p = agent.get_policy(self.state_space)
        u = self.get_state_values_from_policy(p, eps=eps)
        v = self.state_values
        
        assert u.keys() == v.keys()
        
        d = np.array([v[k] - u[k] for k in u])
        assert np.all(d > -1e-3)
        
        return np.sum(d)
    
    def get_policy(self, _states):
        return self.policy


class PolicyAgent(object):
    def __init__(self, action_space,states,gamma,nb_iter,epsilon):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = gamma
        self.states = states
        self.nb_iter = nb_iter
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        return self.Pi[self.states[gridworld.GridworldEnv.state2str(observation)]]

    def fit(self,statedic, mdp):
        diff_value = []
        self.V = np.random.random(len(statedic))
        self.Pi = np.random.randint(0,self.action_space.n,len(statedic))
        nb_iter = self.nb_iter
        epsilon = self.epsilon
        nb_etape_totale = 0
        nb_etape_value = 0
        for iter_en_cours in range(nb_iter):
            #Mise a jours de V
            for _ in range(nb_iter):
                ancien_V = np.copy(self.V)
                new_V = np.copy(self.V)
                for cle_s,valeur_s in statedic.items():
                    if(cle_s not in mdp.keys()):continue
                    somme = 0
                    action_a_faire = self.Pi[valeur_s]
                    list_succ = mdp[cle_s][action_a_faire]
                    for noeud in list_succ:
                        somme +=noeud[0]*( noeud[2] + self.gamma*self.V[valeur_s])
                    new_V[valeur_s] = somme
                self.V = new_V
                nb_etape_value+=1
                if(ancien_V is not None):
                    diff_value.append(np.linalg.norm(ancien_V-self.V))
                if(ancien_V is not None and np.linalg.norm(ancien_V-self.V) < epsilon): break
            #Mise a jours de Pi
            ancien_Pi = np.copy(self.Pi)
            for cle_s,valeur_s in statedic.items():
                if(cle_s not in mdp.keys()): continue
                li = np.zeros(self.action_space.n)
                for a in range(self.action_space.n):
                    somme = 0
                    list_succ = mdp[cle_s][a]
                    for noeud in list_succ:
                        entier_noeud = statedic[noeud[1]]
                        somme +=noeud[0]*( noeud[2] + self.gamma*self.V[entier_noeud])
                    li[a] = somme
                self.Pi[valeur_s] = np.argmax(li)
            nb_etape_totale+=1
            if(ancien_Pi is not None and np.array_equal(self.Pi, ancien_Pi) ):
                print("La politique n'a pas changé entre les deux itérations donc on suppose qu'elle est devenue stable")
                break
        return nb_etape_totale,nb_etape_value,diff_value


class ValueIterationAgentV2:
    def __init__(self, state_space, _action_space, p, gamma=.99):
        self.policy = None
        self.state_values = None
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.last_state = None
    
    def _value_iteration(self, eps):
        current_state_values = self.state_values
        new_state_values = {}
        diff = eps
        all_diff = []
        while diff >= eps:
            for state in self.state_space:
                state_value = -np.inf
                
                for action in self.p[state].keys():
                    value = 0
                    for proba, state2, reward, done in self.p[state][action]:
                        val = reward
                        if not done:
                            val += self.gamma * current_state_values[state2]
                        val *= proba
                        
                        value += val
                    
                    state_value = max(state_value, value)
                
                new_state_values[state] = state_value
            
            diff = 0
            for state in self.state_space:
                diff += (new_state_values[state] - current_state_values[state]) ** 2
                current_state_values[state] = new_state_values[state]
            diff = np.sqrt(diff)
            all_diff.append(diff)
        
        return new_state_values,all_diff
    
    def _update_policy(self):
        """meme code que policy_iteration"""
        new_policy = {}
        for state in self.state_space:
            best_action = None
            max_value = -np.inf
            
            for action in self.p[state].keys():
                s = 0
                for proba, state2, reward, done in self.p[state][action]:
                    val = reward
                    if not done:
                        val += self.gamma * self.state_values[state2]
                    val *= proba
                    s += val
                
                if s > max_value:
                    best_action = action
                    max_value = s
            
            new_policy[state] = best_action
        
        return new_policy
    
    def compute_best_policy(self, eps=1e-6):
        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in self.state_space}
        
        self.state_values,all_diff = self._value_iteration(eps)
        
        self.policy = self._update_policy()

        return len(all_diff),all_diff
    
    def act(self):
        return self.policy[self.last_state]
    
    def reset(self, obs):
        self.last_state = obs
    
    def get_result(self, obs, _reward, _done):
        self.last_state = obs


class ValueAgent(object):
    def __init__(self, action_space,states,gamma,nb_iter,epsilon):
        self.action_space = action_space
        self.V = None
        self.Pi = None
        self.gamma = gamma
        self.states = states
        self.nb_iter = nb_iter
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        return self.Pi[self.states[gridworld.GridworldEnv.state2str(observation)]]

    def fit(self,statedic, mdp):
        diff_value = []
        self.V = np.random.random(len(statedic))
        self.Pi = np.random.randint(0,self.action_space.n,len(statedic))
        nb_iter = self.nb_iter
        epsilon = self.epsilon


        nb_etape_value = 0
        #Mise a jours de V
        for _ in range(nb_iter):
            new_V = self.V.copy()
            ancien_V = np.copy(self.V)
            for cle_s,valeur_s in statedic.items():
                if(cle_s not in mdp.keys()):continue
                maxi = -99999
                for action_a_faire in range(self.action_space.n):
                    somme = 0
                    list_succ = mdp[cle_s][action_a_faire]
                    for noeud in list_succ:
                        if(noeud[3]):
                            somme+=noeud[0]*noeud[2]
                        else:
                            somme +=noeud[0]*( noeud[2] + self.gamma*self.V[valeur_s])
                    if(somme> maxi):
                        maxi=somme
                new_V[valeur_s] = maxi
            self.V = new_V
            nb_etape_value+=1
            if(ancien_V is not None):
                    diff_value.append(np.linalg.norm(ancien_V-self.V))
            if(ancien_V is not None and np.linalg.norm(ancien_V-self.V) < epsilon):
                print(np.linalg.norm(ancien_V-self.V))
                break
        
        #Mise a jours de Pi
        for cle_s,valeur_s in statedic.items():
            if(cle_s not in mdp.keys()): continue
            li = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                somme = 0
                list_succ = mdp[cle_s][a]
                for noeud in list_succ:
                    entier_noeud = statedic[noeud[1]]
                    somme +=noeud[0]*( noeud[2] + self.gamma*self.V[entier_noeud])
                li[a] = somme
            self.Pi[valeur_s] = np.argmax(li)
            
        return nb_etape_value,diff_value


def test_policy_iteration(num_plan):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan"+str(num_plan)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    #env.render()  # permet de visualiser la grille du jeu
    #env.render(mode="human") #visualisation sur la console

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    # Execution avec un Agent
    #agent = PolicyAgent(env.action_space,statedic,gamma=0.99,nb_iter=10000,epsilon=1e-8)
    agent = PolicyIterationAgentV2(mdp.keys(), env.action_space, mdp, gamma=0.99)
    tmps_debut = time.time()
    nb_etape_totale,nb_etape_value,diff_value = agent.compute_best_policy(eps=1e-6)
    #nb_etape_totale,nb_etape_value,diff_value = agent.fit(statedic,mdp)
    #print("Résultat policy iteration :")
    #print("L'entrainement à pris",time.time()-tmps_debut," secondes pour converger")
    print(time.time()-tmps_debut)
    #print("Il a fallu",nb_etape_totale,"étapes de mise a jours de politique et",nb_etape_value," étapes de mise a jours de la value function")
    plt.clf()
    plt.title("Courbe de distance entre V_t et V_t+1 pour Policy iteration")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Distance")
    plt.plot(diff_value)
    #plt.show()

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 100
    reward = 0
    done = False
    rsum = 0
    somme_all_episode = 0
    FPS = 0.0001
    for i in range(episode_count):
        ####BOULCE KEYVAN#########
        obs = env.state2str(env.reset())
        agent.reset(obs)
        ###FIN BOUCLE KEYVAN#####
        #obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            #####BOUCLE ACTION KEYVAN##############
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            obs = env.state2str(obs)
            agent.get_result(obs, reward, done)
            rsum += reward
            #####FIN BOUCLE ACTION KEYVAN##############
            """
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            """
            if done:
                somme_all_episode+=rsum
                #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    #print("La reward moyenne sommer sur les episodes est de ",somme_all_episode/episode_count)
    print(somme_all_episode/episode_count)
    env.close()



def test_value_iteration(num_carte):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan"+str(num_carte)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    #env.render()  # permet de visualiser la grille du jeu
    #env.render(mode="human") #visualisation sur la console
   
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    # Execution avec un Agent
    #agent = ValueIteration(env.action_space,statedic,gamma=0.99,nb_iter=100000,epsilon=1e-15)
    agent = ValueIterationAgentV2(mdp.keys(), env.action_space, mdp, gamma=0.99)
    nb_etape_value,diff_value = agent.compute_best_policy(eps=1e-6)
    tmps_debut = time.time()
    #nb_etape_value,diff_value = agent.fit(statedic,mdp)
    #print("Résultat Value iteration :")
    #print("L'entrainement à pris",time.time()-tmps_debut," secondes pour converger")
    print(time.time()-tmps_debut)
    #print("Il a fallu",nb_etape_value," étapes de mise a jours de la value function")
    plt.clf()
    plt.title("Courbe de distance entre V_t et V_t+1 pour Value iteration")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Distance")
    plt.plot(diff_value)
    #plt.show()
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 100
    reward = 0
    done = False
    rsum = 0
    somme_all_episode = 0
    FPS = 0.0001
    for i in range(episode_count):
        ####BOULCE KEYVAN#########
        obs = env.state2str(env.reset())
        agent.reset(obs)
        ###FIN BOUCLE KEYVAN#####
        #obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            #####BOUCLE ACTION KEYVAN##############
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            obs = env.state2str(obs)
            agent.get_result(obs, reward, done)
            rsum += reward
            #####FIN BOUCLE ACTION KEYVAN##############
            """
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            """
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                somme_all_episode+=rsum
                #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    #print("La reward moyenne sommer sur les episodes est de ",somme_all_episode/episode_count)
    print(somme_all_episode/episode_count)
    env.close()


if __name__ == '__main__':
    """
    for k in range(9):
        print(k)
        test_policy_iteration(k)
        test_value_iteration(k)
    """
    test_policy_iteration(9)
    test_value_iteration(9)
    