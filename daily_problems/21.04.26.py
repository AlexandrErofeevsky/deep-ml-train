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
