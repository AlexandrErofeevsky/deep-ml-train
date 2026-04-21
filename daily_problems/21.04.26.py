"""
Estimate Action Values Using Sample Averaging
 - Easy
 - Reinforcement Learning

In reinforcement learning, an agent must estimate how good each available action is based on past experience.
One of the most fundamental approaches is sample averaging: the estimated value of an action is simply the mean of all rewards received when that action was selected.

Implement a function that processes a sequence of action-reward pairs and computes the estimated value for each of k possible actions (labeled 0 to k-1).
If an action has never been selected, its estimated value should remain 0.0.

Parameters:

    k: integer, the number of possible actions (labeled 0 to k-1)
    actions: list of integers, the action selected at each time step
    rewards: list of floats, the reward received at each corresponding time step

Returns:

    A tuple (Q, N) where:
        Q is a list of k floats representing the estimated value of each action, each rounded to 4 decimal places
        N is a list of k integers representing how many times each action was selected

Example:
    Input:
        k=2, actions=[0, 1, 0, 1, 0], rewards=[1.0, 2.0, 3.0, 4.0, 5.0]
    Output:
        Q = [3.0, 3.0], N = [3, 2]
Reasoning:
    Action 0 was selected at steps 0, 2, and 4 with rewards 1.0, 3.0, and 5.0.
    Its estimated value is (1.0 + 3.0 + 5.0) / 3 = 3.0.
    Action 1 was selected at steps 1 and 3 with rewards 2.0 and 4.0.
    Its estimated value is (2.0 + 4.0) / 2 = 3.0.
"""


from collections import defaultdict


def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.

    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step

    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    means = defaultdict(list)
    counter = defaultdict(int)
    for act, rew in zip(actions, rewards):
        means[act].append(rew)
        counter[act] += 1
    q = []
    n = []
    for i in range(k):
        n.append(counter[i])
        if means[i]:
            q.append(round(sum(means[i]) / len(means[i]), 4))
        else:
            q.append(0.)
    return q, n
