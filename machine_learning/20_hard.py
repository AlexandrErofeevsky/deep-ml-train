"""
Decision Tree Learning
 - Hard
 - Machine Learning

Write a Python function that implements the decision tree learning algorithm for classification.
The function should use recursive splitting based on entropy and information gain to build a decision tree.
It should take a list of examples (each example is a dict of attribute-value pairs)
and a list of attribute names as input, and return a nested dictionary representing the decision tree.

Tie-Breaking Rules:
    If multiple attributes have equal information gain,
    choose the one that appears first in the attributes list.
    If a leaf node has equal counts of different classes,
    return the class that comes first alphabetically.
    Process attribute values in sorted order to ensure consistent tree structure.

Example:
    Input:
        examples = [
            {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
            {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
            {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
            {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
            {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
            {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
            {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
        ],
        attributes = ['Outlook', 'Wind'],
        target_attr = 'PlayTennis'
    Output:
        {
            'Outlook': {'
                Overcast': 'Yes',
                'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
                'Sunny': 'No'
            }
        }
Reasoning:
    The algorithm first calculates information gain for each attribute.
    'Outlook' has the highest gain, so it becomes the root.
    For 'Overcast', all outcomes are 'Yes' (pure leaf).
    For 'Sunny', all outcomes are 'No' (pure leaf - no need to split further).
    For 'Rain', the data has mixed labels, so it recurses and splits on 'Wind'.
"""


import math
from collections import Counter, defaultdict


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    counter = Counter(labels)
    n = len(labels)
    return - sum(val / n * math.log2(val / n) for val in counter.values())


def calculate_information_gain(
    examples: list[dict], attr: str, target_attr: str,
) -> float:
    """Calculate the information gain of splitting on attr."""
    n = len(examples)
    labels_attr = defaultdict(list)
    labels = []

    for ex in examples:
        labels_attr[ex[attr]].append(ex[target_attr])
        labels.append(ex[target_attr])

    return calculate_entropy(labels) - sum(
        len(val) / n * calculate_entropy(val) for val in labels_attr.values()
    )


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [ex[target_attr] for ex in examples]
    counter = Counter(labels)

    max_count = max(counter.values())

    return sorted(
        [item for item, count in counter.items() if count == max_count]
    )[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict | str:
    """Build a decision tree using the ID3 algorithm."""
    if not attributes:
        return majority_class(examples, target_attr)
    max_ig = -1
    max_ig_attr = ""
    for attr in attributes:
        ig = calculate_information_gain(
            examples, attr, target_attr,
        )
        if ig > max_ig:
            max_ig = ig
            max_ig_attr = attr

    if len({ex[target_attr] for ex in examples}) == 1:
        return majority_class(examples, target_attr)

    attr_dict = defaultdict(list)

    for ex in examples:
        attr_dict[ex[max_ig_attr]].append(ex)

    attrs = [attr for attr in attributes if attr != max_ig_attr]

    result = {}
    for key in sorted(attr_dict.keys()):
        result[key] = learn_decision_tree(attr_dict[key], attrs, target_attr)

    return {max_ig_attr: result}
