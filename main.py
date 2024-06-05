import random
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login

def setup_model(model_name, token):
    login(token=token)
    llm = LLM(model=model_name, gpu_memory_utilization=0.5, enforce_eager=True)
    return llm

def load_dataset_by_name(dataset_name, subset, num_samples):
    if dataset_name == 'gsm8k':
        dataset = load_dataset('gsm8k', subset)
        train_set = dataset['train'].select(range(num_samples))
        test_set = dataset['test'].select(range(num_samples))
    elif dataset_name == 'bbh':
        dataset = load_dataset('lukaemon/bbh', subset)
        train_set = dataset['test'].select(range(num_samples)) # Has no train set
        test_set = dataset['test'].select(range(num_samples))
    elif dataset_name == 'mmlu':
        dataset = load_dataset('cais/mmlu', subset, trust_remote_code=True)
        train_set = dataset['auxiliary_train'].select(range(num_samples))
        test_set = dataset['test'].select(range(num_samples))
    else:
        raise ValueError("Unsupported dataset name")
    return train_set, test_set

def create_prompt_gsm8k(example, in_context_examples):
    prompt = ""
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += "Give me the answer for the following question:\n"
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
    prompt = "Give me the answer for the following questions. Give your answer as an index from 0 to 3 for the given choices. Here are a few examples:\n"
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nChoices: {ex['choices']}\nA: {ex['answer']}\n\n"
    prompt += "Now Answer this question:\n"
    prompt += f"Q: {example['question']}\nChoices: {example['choices']}\nA: "
    return prompt

def evaluate_model(llm, dataset_name, dataset, in_context_examples, create_prompt_fn, max_length=256, output_file=None):
    correct = 0
    total = 0

    prompts = [create_prompt_fn(example, in_context_examples) for example in dataset]
    sampling_params = SamplingParams(max_tokens=max_length)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, example in enumerate(dataset):
        generated_text = outputs[idx].outputs[0].text

        if dataset_name == 'bbh':
            correct_answer = example["target"].strip()
        elif dataset_name == 'mmlu':
            correct_answer = str(example["answer"])
        else:
            correct_answer = example["answer"].split('####')[-1].strip()

        is_correct = correct_answer in generated_text

        total += 1
        if is_correct:
            correct += 1

        if output_file:
            output_file.write(f"Q: {example.get('question', example.get('input', ''))}\n")
            output_file.write(f"Generated A: {generated_text}\n")
            output_file.write(f"Correct A: {correct_answer}\n")
            output_file.write(f"Is Correct: {is_correct}\n")
            output_file.write("\n")

    accuracy = correct / total
    return accuracy

def extract_answer(generated_text):
    match = re.search(r'(\d+)', generated_text)
    return match.group(1) if match else None

def generate_random_orderings(data, num_orderings=10):
    orderings = []
    for _ in range(num_orderings):
        order = list(range(len(data)))
        random.shuffle(order)
        orderings.append(order)
    return orderings

# Configuration
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Change to "meta-llama/Llama-2-13b-hf" as needed
model_name = "meta-llama/Llama-2-13b-hf"
token = "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"
dataset_name = "bbh"  # Change this to "lukaemon/bbh" or "cais/mmlu" as needed
subset = "main"  # For gsm8k use 'main'; for MMLU use 'all'
if dataset_name == "mmlu": subset = "all"
elif dataset_name == "bbh": subset = "boolean_expressions"
num_samples = 100

# Setup
llm = setup_model(model_name, token)
train_set, test_set = load_dataset_by_name(dataset_name, subset, num_samples)

# Select in-context examples from training data and test data from the test set
in_context_data = train_set.select(range(8))
test_data = test_set.select(range(50))

# Create random orderings
random_orderings = generate_random_orderings(in_context_data, num_orderings=10)

# Evaluate and write results
results = []
with open(f"{dataset_name}_evaluation_results.txt", "w") as f, open(f"{dataset_name}_outputs.txt", "w") as output_file:
    for i, ordering in enumerate(random_orderings):
        ordered_examples = [in_context_data[idx] for idx in ordering]
        if dataset_name == 'gsm8k':
            create_prompt_fn = create_prompt_gsm8k
        elif dataset_name == 'bbh':
            create_prompt_fn = create_prompt_bbh
        elif dataset_name == 'mmlu':
            create_prompt_fn = create_prompt_mmlu

        accuracy = evaluate_model(llm, dataset_name, test_data, ordered_examples, create_prompt_fn, output_file=output_file)
        results.append(accuracy)
        f.write(f"Ordering {i+1}: Accuracy = {accuracy}\n")
        print(f"Ordering {i+1}: Accuracy = {accuracy}")

for i, accuracy in enumerate(results):
    print(f"Ordering {i+1}: Accuracy = {accuracy}")
