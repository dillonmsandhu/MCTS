import copy
import math
import random
from time import sleep

class Node(object):

    def __init__(self):
        self.visits = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0

    def expanded(self): 
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def expand(self,state,legal_actions,done):
        """Initialize all children of a given leaf node"""
        if not done:
            for action in legal_actions:
                self.children[action] = Node()

def tree_policy(legal_actions):
    return random.choice(legal_actions)

def backpropagate(search_path, value, discount):
    """Update G[t:] with a sampled value from the rollout"""
    # Why not roll all the way up the game until the first state, instead of just the 
    # first state in the tree? <- most likely memory concerns
    for node in reversed(search_path):
        node.visits+=1
        value = node.reward + discount * value 
        node.value_sum += value
        
def select_child(node): #-> Policy over the expanded part of the tree
    """Select a random child node with the max upper confidence bound. Returns action, node"""
    highest = max(ucb_score(node, child) for action, child in node.children.items())
    best_children = [(action, child) for action, child in node.children.items() if ucb_score(node,child)==highest]
    return random.choice(best_children)

def ucb_score(parent,child):
    if child.visits==0:
        return math.inf
    value_child = child.value_sum/child.visits
    confidence = math.sqrt(2*math.log(parent.visits)/child.visits)
    return value_child + confidence

def select_action(root): # final choice after running MCTS
    """Picks the most explored child node"""
    max_visits = max(child.visits for action, child in root.children.items())
    most_explored = [action for action, child in root.children.items() if child.visits==max_visits]
    return random.choice(most_explored)

def run_mcts(root,env,discount,num_simulations,randomize_env=False):
    
    for _ in range(num_simulations):
        simulation_env = copy.deepcopy(env) # Note this creates problems rendering. 
        if randomize_env: simulation_env.seed() # Do not allow the search to access the exact same random seed.
        actions = list(range(env.action_space.n))
        node = root
        search_path = [node]        
        
        # Expand until a previously unexpanded (leaf) node is reached
        while node.expanded():
            action, node = select_child(node) # new node associated with the action.
            search_path.append(node) 
            state, node.reward, done, _ = simulation_env.step(action)
        parent = search_path[-2]
        node.expand(state,actions,done) #initializes children
        
        # Estimate the value of the leaf node when following tree_policy
        sim_val = 0 # value from the expanded node onwards
        sim_len = 0 # length of episode following expanded node
        while not done:
            action = tree_policy(actions)
            _, rew, done, _ = simulation_env.step(action)
            sim_val += rew*discount**sim_len # r_1+gamma*r_2 + gamma^2*r_3 ...
            sim_len +=1        
        simulation_env.close()
        backpropagate(search_path, sim_val, discount)

def play_episode(env,num_simulations,render=False,randomize_env = False, discount=1):
    state = env.reset()
    done = False
    ep_rew,ep_len = 0,0
    while not done:
        root = Node()
        actions = list(range(env.action_space.n))
        root.expand(state,actions,done=False)
        run_mcts(root,env,discount,num_simulations,randomize_env)
        action = select_action(root)
        state, r, done, _ = env.step(action)
        ep_rew += r
        ep_len += 1
        if render: 
            sleep(1)
            env.render()
    if render:
        print("Episode finished after {} timesteps with reward {}".format(ep_len,ep_rew))
    return ep_len, ep_rew