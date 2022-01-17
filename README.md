# MCTS
### Monte-Carlo Tree Search for use on gym environments. 

The popular [UCT](http://ggp.stanford.edu/readings/uct.pdf) algorithm. MCTS can be thought of a low-memory representation of a policy, which can be used when a simulator of the environment is available. At each state, MCTS runs simulations and returns an action, which roughly maximizes an estimated Q-value. UCT is a highly-selective best-first search, meaning that it can look quite far ahead, making it useful in games like Chess and Go. In [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far), MCTS acts as a policy-improvement step: improving upon the policy encoded in a neural net, as well as the online-player.

This algorithm is useful when a simulator of the environment is available. Starting with the current state, all actions are simulated. If the next state-action pair has been encountered before, UCT picks the action with the highest upper-confidence bound on Q-value. This is implemented in the function `select_child`. The first time a particular state-action pair is encountered (a leaf node), its Q-value is estimated by randomly picking actions until the simulation ends. This estimated Q-value for the leaf node is incorporated into the Q-values of transitions leading to the leaf node, with appropriate discounting, by the `backpropagate` function. After `num_simulations` runs, the action picked the most is returned.  

There are numerous ways to modify MCTS. In environments with a long or infinite horizon, a statistical estimate of the value function can be used to assess the value of leaf nodes. 

