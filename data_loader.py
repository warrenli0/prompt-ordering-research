import os
from datasets import load_dataset, Dataset
import pandas as pd
import nltk
from collections import defaultdict, Counter
import random

# Ensure that the NLTK punkt tokenizer is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

def load_dataset_by_name(dataset_name, num_samples):
    label_names = None
    if dataset_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main')
        train_set = dataset['train']
        test_set = dataset['test']
        num_classes = 4  # No discrete classes, but select 5 examples
    elif dataset_name == 'bbh':
        dataset = load_dataset('lukaemon/bbh', 'boolean_expressions')
        train_set = dataset['test']  # Has no train set
        test_set = dataset['test']
        num_classes = 2  # Assuming binary classification
        label_names = ['True', 'False']
    elif dataset_name == 'mmlu':
        dataset = load_dataset('cais/mmlu', 'all', trust_remote_code=True)
        train_set = dataset['test']
        test_set = dataset['test']
        num_classes = 4  # MMLU typically has 4 answer choices
        label_names = ['A', 'B', 'C', 'D']

        # Define the subjects you want to keep
        desired_subjects = ['college_mathematics', 'elementary_mathematics', 'high_school_mathematics']

        # Filter the train set for the desired subjects
        train_set = train_set.filter(lambda x: x['subject'] in desired_subjects)

        # Filter the test set for the desired subjects
        test_set = test_set.filter(lambda x: x['subject'] in desired_subjects)

        print(f"Number of examples in filtered train set: {len(train_set)}")
        print(f"Number of examples in filtered test set: {len(test_set)}")

        # dev_subjects = [example['subject'] for example in train_set if 'subject' in example]
        # dev_subject_count = Counter(dev_subjects)

        # test_subjects = [example['subject'] for example in test_set if 'subject' in example]
        # test_subject_count = Counter(test_subjects)

        # print("Dev set subject counts:")
        # for subject, count in dev_subject_count.items():
        #     print(f"{subject}: {count}")

        # print("\nTest set subject counts:")
        # for subject, count in test_subject_count.items():
        #     print(f"{subject}: {count}")
    elif dataset_name == 'ag_news':
        dataset = load_dataset('fancyzhx/ag_news', 'default', trust_remote_code=True)
        train_set = dataset['train']
        test_set = dataset['test']
        num_classes = 4
        label_names = ['World', 'Sports', 'Business', 'Technology']
    elif dataset_name == 'sst2':
        dataset = load_dataset('stanfordnlp/sst2', 'default', trust_remote_code=True)
        train_set = dataset['train']
        test_set = dataset['test']
        num_classes = 2
        label_names = ['negative', 'positive']
    elif dataset_name == 'dbpedia':
        dataset = load_dataset('fancyzhx/dbpedia_14', 'dbpedia_14', trust_remote_code=True)
        train_set = dataset['train']
        test_set = dataset['test']
        num_classes = 14
        label_names = [
            'Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder', 'MeanOfTransportation',
            'Building', 'NaturalPlace', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'WrittenWork'
        ]
    elif dataset_name in ['nyt-topics', 'nyt-locations']:
        return load_custom_dataset(dataset_name, num_samples)
    else:
        raise ValueError("Unsupported dataset name")
    
    # Sort datasets by label and text
    # train_set = train_set.sort('label').sort('content')
    # test_set = test_set.sort('label').sort('content')
    
    # Return None for label_names if not custom dataset
    return train_set, test_set, num_classes, label_names

def load_custom_dataset(dataset_name, num_samples):
    base_path = os.path.join('data', dataset_name.replace('nyt', 'NYT'))
    train_texts = load_text_file(os.path.join(base_path, 'train_text.txt'))
    test_texts = load_text_file(os.path.join(base_path, 'test_text.txt'))
    train_labels = load_label_file(os.path.join(base_path, 'train_label.txt'))
    test_labels = load_label_file(os.path.join(base_path, 'test_label.txt'))
    label_names = load_text_file(os.path.join(base_path, 'label_names.txt'))
    
    num_classes = len(label_names)

    train_data = {'text': train_texts, 'label': train_labels}
    test_data = {'text': test_texts, 'label': test_labels}

    train_set = Dataset.from_pandas(pd.DataFrame(train_data))
    test_set = Dataset.from_pandas(pd.DataFrame(test_data))

    # Sort datasets by label and text
    train_set = train_set.sort('label').sort('text')
    test_set = test_set.sort('label').sort('text')

    return train_set, test_set, num_classes, label_names

def load_test_set(test_set, label_key, num_test_examples, dataset_name=None, seed=42):
    import random
    random.seed(seed)

    if dataset_name == 'gsm8k':
        total_indices = list(range(len(test_set)))
        random.shuffle(total_indices)
        selected_indices = total_indices[:num_test_examples]
        test_data = test_set.select(selected_indices)

        return test_data
    
    num_classes = len(set(test_set[label_key]))
    num_test_examples_per_label = num_test_examples // num_classes

    total_indices = list(range(len(test_set)))
    selected_indices = []

    for label in range(num_classes):
        # Get indices of examples with this label
        label_indices = [i for i, x in enumerate(test_set[label_key]) if x == label]

        # Shuffle the indices
        random.shuffle(label_indices)

        # Determine the number of examples to select
        available_examples = len(label_indices)
        num_examples_to_select = min(num_test_examples_per_label, available_examples)

        # Warn if not enough examples are available
        if available_examples < num_test_examples_per_label:
            print(f"Warning: Only {available_examples} examples available for label {label}, but {num_test_examples_per_label} were requested.")

        # Select the examples
        selected_label_indices = label_indices[:num_examples_to_select]
        selected_indices.extend(selected_label_indices)

    # Now check if total_selected_examples < num_test_examples
    total_selected_examples = len(selected_indices)
    remaining_examples_to_select = num_test_examples - total_selected_examples

    if remaining_examples_to_select > 0:
        # Get indices of examples not already selected
        available_indices = list(set(total_indices) - set(selected_indices))

        if len(available_indices) < remaining_examples_to_select:
            print(f"Warning: Only {len(available_indices)} additional examples available, but {remaining_examples_to_select} were requested.")

        # Shuffle available indices
        random.shuffle(available_indices)

        # Select remaining examples
        additional_indices = available_indices[:remaining_examples_to_select]
        selected_indices.extend(additional_indices)

    # Now select the examples from test_set
    test_data = test_set.select(selected_indices)

    # Shuffle the final test_data
    test_data = test_data.shuffle(seed=42)

    return test_data

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        truncated_lines = [truncate_text(line.strip()) for line in lines]
        return truncated_lines

def load_label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [int(line.strip()) for line in file]

def truncate_text(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Take the first sentence
    first_sentence = sentences[0] if sentences else text
    # Truncate to 50 tokens
    tokens = nltk.word_tokenize(first_sentence)
    truncated_text = ' '.join(tokens[:50])
    return truncated_text

# Pre-sample and shuffle examples per class at the start to speed up selection
def prefilter_and_sample_examples(train_set, num_classes, num_incontext_examples, seed=42):
    examples_per_class = num_incontext_examples // num_classes
    sampled_data = defaultdict(list)
    
    # Pre-filter and shuffle examples for each class, and select the required number of examples
    for label in range(num_classes):
        class_examples = train_set.filter(lambda example: example['label'] == label)
        sampled_data[label] = class_examples.shuffle(seed=seed).select(range(examples_per_class))
    
    return sampled_data

# Select in-context examples (just combine the pre-sampled examples)
def select_in_context_examples(sampled_data, num_classes):
    in_context_examples = []
    for label in range(num_classes):
        in_context_examples.extend(sampled_data[label])
    return in_context_examples

# Pre-sample and shuffle more examples per class at the start to allow random selection in each run
def prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, dataset_name=None, seed=42):
    sampled_data = defaultdict(list)

    if dataset_name == 'gsm8k':
        num_to_select = min(total_examples_per_class * num_classes, len(train_set))
        sampled_data[0] = train_set.shuffle(seed=seed).select(range(num_to_select))
        return sampled_data

    # Check the keys in the first example to determine the label key
    first_example = train_set[0]
    label_key = 'label' if 'label' in first_example else 'answer'

    # Pre-filter and shuffle examples for each class, and select up to the available number of examples
    for label in range(num_classes):
        class_examples = train_set.filter(lambda example: example[label_key] == label)
        num_available = len(class_examples)

        # Ensure we don't try to select more examples than available
        num_to_select = min(total_examples_per_class, num_available)

        sampled_data[label] = class_examples.shuffle(seed=seed).select(range(num_to_select))
    
    return sampled_data

# Select in-context examples by randomly sampling from the pre-sampled data
def select_in_context_examples_multiple(sampled_data, num_classes, num_incontext_examples, dataset_name=None, seed=None):
    random.seed(seed)
    examples_per_class = num_incontext_examples // num_classes
    in_context_examples = []

     # Handle special case for gsm8k (no class labels)
    if dataset_name == 'gsm8k':
        num_to_select = min(num_incontext_examples, len(sampled_data[0]))
        in_context_examples = sampled_data[0].shuffle(seed=seed).select(range(num_to_select))
        return list(in_context_examples)
    
    for label in range(num_classes):
        # Convert the dataset object to a list for sampling
        class_examples = list(sampled_data[label])
        
        # Ensure that we sample from the list of examples
        sampled_examples = random.sample(class_examples, examples_per_class)
        in_context_examples.extend(sampled_examples)
    
    return in_context_examples

# Select equal number of examples per class for in-context examples
# def select_in_context_examples(train_set, num_classes, num_incontext_examples):
#     examples_per_class = num_incontext_examples // num_classes
#     in_context_examples = []

#     for label in range(num_classes):
#         class_examples = train_set.filter(lambda example: example['label'] == label)
#         selected_examples = class_examples.shuffle(seed=42).select(range(examples_per_class))
#         in_context_examples.extend(selected_examples)

#     return in_context_examples
