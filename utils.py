import random
import re
import numpy as np

def extract_answer(generated_text):
    match = re.search(r'(\d+)', generated_text)
    return match.group(1) if match else None

def generate_random_orderings(data, num_orderings=10, seed=None):
    if seed is not None:
        random.seed(seed)

    orderings = []
    for _ in range(num_orderings):
        order = list(range(len(data)))
        random.shuffle(order)
        shuffled_order = [order.index(i) for i in range(len(data))]
        orderings.append(order)
    return orderings

def reorder_list(original_list, ordering):
    shuffled_list = [None] * len(original_list)
    for idx, new_idx in enumerate(ordering):
        shuffled_list[new_idx] = original_list[idx]
    return shuffled_list

# Swap first example with every other example
def generate_swap_orderings(data):
    orderings = []
    n = len(data)
    for i in range(n):
        if i == 0:
            orderings.append((list(range(n)), list(range(n))))
        else:
            new_order = list(range(n))
            new_order[0], new_order[i] = new_order[i], new_order[0]
            shuffled_order = [new_order.index(j) for j in range(n)]
            orderings.append((new_order, shuffled_order))
    return orderings

# Shuffle within each label and return the list
def shuffle_within_label(data, seed=42):
    label_to_examples = {}
    for example in data:
        label = example['label']
        if label not in label_to_examples:
            label_to_examples[label] = []
        label_to_examples[label].append(example)

    # Shuffle examples within each label using the provided seed
    for label in label_to_examples:
        random.Random(seed).shuffle(label_to_examples[label])

    # Flatten the dictionary back into a list
    shuffled_data = []
    for label in sorted(label_to_examples.keys()):
        shuffled_data.extend(label_to_examples[label])

    return shuffled_data

# Randomly shuffle the order of the labels while keeping the order within each label intact
def randomize_label_order(data, seed=42):
    label_to_examples = {}
    for example in data:
        label = example['label']
        if label not in label_to_examples:
            label_to_examples[label] = []
        label_to_examples[label].append(example)

    # Shuffle labels
    labels = list(label_to_examples.keys())
    random.Random(seed).shuffle(labels)

    # Rebuild the dataset with the shuffled labels
    shuffled_data = []
    for label in labels:
        shuffled_data.extend(label_to_examples[label])

    return shuffled_data

# Shuffle across the entire set
def shuffle_entire_set(data, seed=42):
    random.Random(seed).shuffle(data)
    return data