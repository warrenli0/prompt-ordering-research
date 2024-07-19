import random
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login
import pandas as pd

def setup_model(model_name, token):
    login(token=token)
    llm = LLM(model=model_name, gpu_memory_utilization=0.9, enforce_eager=True)
    return llm

def load_dataset_by_name(dataset_name, num_samples):
    if dataset_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main')
        train_set = dataset['train'].select(range(num_samples))
        test_set = dataset['test'].select(range(num_samples))
    elif dataset_name == 'bbh':
        dataset = load_dataset('lukaemon/bbh', 'boolean_expressions')
        train_set = dataset['test'].select(range(num_samples)) # Has no train set
        test_set = dataset['test'].select(range(num_samples))
    elif dataset_name == 'mmlu':
        dataset = load_dataset('cais/mmlu', 'all', trust_remote_code=True)
        train_set = dataset['auxiliary_train'].select(range(num_samples))
        test_set = dataset['test'].select(range(num_samples))
    elif dataset_name == 'ag_news':
        dataset = load_dataset('fancyzhx/ag_news', 'default', trust_remote_code=True)
        train_set = dataset['train']
        test_set = dataset['test']
    elif dataset_name == 'sst2':
        dataset = load_dataset('stanfordnlp/sst2', 'default', trust_remote_code=True)
        train_set = dataset['train'].select(range(num_samples))
        test_set = dataset['validation'].select(range(num_samples))
    elif dataset_name == 'dbpedia':
        dataset = load_dataset('fancyzhx/dbpedia_14', 'dbpedia_14', trust_remote_code=True)
        train_set = dataset['train']
        test_set = dataset['test']
    else:
        raise ValueError("Unsupported dataset name")
    return train_set, test_set

def create_prompt_gsm8k(example, in_context_examples):
    prompt = "Given the answer to the following questions:"
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += "Answer to the following question:\n"
    prompt += f"Q: {example['question']}\nA: "
    return prompt

def create_prompt_bbh(example, in_context_examples):
    prompt = ""
    for ex in in_context_examples:
        prompt += f"Q: {ex['input']}\nA: {ex['target']}\n\n"
    prompt += "Give me the answer for the following question:\n"
    prompt += f"Q: {example['input']}\nA: "
    return prompt

def create_prompt_mmlu(example, in_context_examples):
    prompt = "Given are the answer for the following questions as an index from 0 to 3 for the given choices:\n"
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nChoices: {ex['choices']}\nA: {ex['answer']}\n\n"
    prompt += "Now answer this question:\n"
    prompt += f"Q: {example['question']}\nChoices: {example['choices']}\nA: "
    return prompt

def create_prompt_agnews(example, in_context_examples):
    prompt = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
    d = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
    for ex in in_context_examples:
        prompt += f"Article: {ex['text']}\nAnswer: {d[ex['label']][0]}\n\n"
    prompt += "Classify this text:\n"
    prompt += f"Article: {example['text']}\nAnswer:"
    return prompt

def create_prompt_sst2(example, in_context_examples):
    prompt = "Classify the sentiment of the following sentences as either positive or negative.\n\n"
    sentiment_dict = {0: 'negative', 1: 'positive'}
    for ex in in_context_examples:
        prompt += f"Sentence: {ex['sentence']}\nSentiment: {sentiment_dict[ex['label']]}\n\n"
    prompt += "Classify this sentence:\n"
    prompt += f"Sentence: {example['sentence']}\nSentiment:"
    return prompt

def create_prompt_dbpedia(example, in_context_examples):
    prompt = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
    label_dict = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Athlete'], 4: ['Politician'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
    for ex in in_context_examples:
        prompt += f"Article: {ex['content']}\nCategory: {label_dict[ex['label']][0]}\n\n"
    prompt += "Classify this article:\n"
    prompt += f"Article: {example['content']}\nCategory:"
    return prompt

def evaluate_model(llm, dataset_name, dataset, in_context_examples, create_prompt_fn, max_length=256, output_file=None):
    correct = 0
    total = 0
    correctness = []

    prompts = [create_prompt_fn(example, in_context_examples) for example in dataset]
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, example in enumerate(dataset):
        generated_text = outputs[idx].outputs[0].text

        if dataset_name == 'bbh':
            correct_answer = example["target"].strip()
        elif dataset_name == 'mmlu':
            correct_answer = str(example["answer"])
        elif dataset_name == 'ag_news':
            inv = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3}
            correct_answer = str(example["label"])
            generated_text = str(inv[generated_text.strip()])
        elif dataset_name == 'sst2':
            inv = {'negative': 0, 'positive': 1}
            correct_answer = str(example["label"])
            generated_text = str(inv[generated_text.strip()])
        elif dataset_name == 'dbpedia':
            inv = {'Company': 0, 'School': 1, 'Artist': 2, 'Athlete': 3, 'Politician': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
            # inv = {
            #     'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 
            #     'MeanOfTransportation': 5, 'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal:': 9, 
            #     'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13
            # }
            correct_answer = str(example["label"])
            generated_text = str(inv.get(generated_text.strip(), -1))
        else:
            correct_answer = example["answer"].split('####')[-1].strip()

        is_correct = correct_answer in generated_text
        correctness.append(is_correct)

        total += 1
        if is_correct:
            correct += 1

        if output_file and idx < 5:
            if idx == 0: output_file.write(f"Prompt: {prompts[idx]}\n")
            output_file.write(f"Q: {example.get('text', example.get('input', ''))}\n")
            output_file.write(f"Generated A:{generated_text}\n")
            output_file.write(f"Correct A: {correct_answer}\n")
            output_file.write(f"Is Correct: {is_correct}\n")
            output_file.write("\n")

    accuracy = correct / total
    return accuracy, correctness

def extract_answer(generated_text):
    match = re.search(r'(\d+)', generated_text)
    return match.group(1) if match else None

def generate_random_orderings(data, num_orderings=10):
    orderings = []
    for _ in range(num_orderings):
        order = list(range(len(data)))
        random.shuffle(order)
        shuffled_order = [order.index(i) for i in range(len(data))]
        orderings.append((order, shuffled_order))
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

# Configuration
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Change to "meta-llama/Llama-2-13b-hf" as needed
model_name = "gpt2-xl"
token = "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"
dataset_name = "ag_news"  # "gsm8k" "lukaemon/bbh" or "cais/mmlu" as needed
num_samples = 100

# Setup
llm = setup_model(model_name, token)
train_set, test_set = load_dataset_by_name(dataset_name, num_samples)
train_set = train_set.shuffle()
test_set = test_set.shuffle()

# Select in-context examples from training data and test data from the test set
in_context_data = train_set.select(range(8)) # Can use more examples for ag_news
test_data = test_set.select(range(50))

# Create random orderings
random_orderings = generate_random_orderings(in_context_data, num_orderings=10)
swap_orderings = generate_swap_orderings(in_context_data)

# Evaluate and write results
results = []
correctness_dict = {}
with open(f"{dataset_name}_evaluation_results.txt", "w") as f, open(f"{dataset_name}_outputs.txt", "w") as output_file:
    for i, (ordering, shuffled_order) in enumerate(swap_orderings): # or random_orderings
        ordered_examples = [in_context_data[idx] for idx in ordering]
        if dataset_name == 'gsm8k':
            create_prompt_fn = create_prompt_gsm8k
        elif dataset_name == 'bbh':
            create_prompt_fn = create_prompt_bbh
        elif dataset_name == 'mmlu':
            create_prompt_fn = create_prompt_mmlu
        elif dataset_name == 'ag_news':
            create_prompt_fn = create_prompt_agnews
        elif dataset_name == 'sst2':
            create_prompt_fn = create_prompt_sst2
        elif dataset_name == 'dbpedia':
            create_prompt_fn = create_prompt_dbpedia

        output_file.write(f"Ordering {i}:\n")
        accuracy, correctness = evaluate_model(llm, dataset_name, test_data, ordered_examples, create_prompt_fn, output_file=output_file)
        results.append(accuracy)
        correctness_dict[i] = correctness
        f.write(f"Ordering {i+1}: Accuracy = {accuracy}\n")
        f.write(f"Shuffled Order: {shuffled_order}\n\n")
        print(f"Ordering {i+1}: Accuracy = {accuracy}")

for i, accuracy in enumerate(results):
    print(f"Ordering {i+1}: Accuracy = {accuracy}")

correctness_df = pd.DataFrame(correctness_dict)
correctness_df.to_csv(f"{dataset_name}_correctness_comparison.csv", index=False)
