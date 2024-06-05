import random
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Llama-2-13b-hf"
token = "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"

login(token=token)

mmlu = load_dataset('cais/mmlu', 'all', trust_remote_code=True)['test'].select(range(100))

llm = LLM(model=model_name, gpu_memory_utilization=0.8, enforce_eager=True)

def create_prompt(example, in_context_examples):
    prompt = ""
    prompt += "Give me the answer for the following questions. Give your answer as an index from 0 to 3 for the given choices. Here are a few examples:\n"
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nChoices: {ex['choices']}\nA: {ex['answer']}\n\n"
    prompt += "Now Answer this question:\n"
    prompt += f"Q: {example['question']}\nChoices: {example['choices']}\nA: "
    return prompt

def evaluate_model(llm, dataset, in_context_examples, max_length=256, output_file=None):
    correct = 0
    total = 0

    prompts = []
    for example in dataset:  # Create all prompts first
        prompt = create_prompt(example, in_context_examples)
        prompts.append(prompt)

    sampling_params = SamplingParams(max_tokens=max_length)

    # Generate outputs for all prompts in a batch
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, example in enumerate(dataset):
        generated_text = outputs[idx].outputs[0].text
        correct_answer = str(example["answer"])

        is_correct = correct_answer in generated_text

        total += 1
        if is_correct:
            correct += 1

        # Write the outputs to the file if provided
        if output_file:
            output_file.write(f"Prompt: {prompt}")
            output_file.write(f"Q: {example['question']}\n")
            output_file.write(f"Choices: {example['choices']}\n")
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

in_context_data = mmlu.select(range(8))
test_data = mmlu.select(range(50, 100))

random_orderings = generate_random_orderings(in_context_data, num_orderings=10)

results = []
with open("mmlu_evaluation_results.txt", "w") as f, open("mmlu_outputs.txt", "w") as output_file:
    for i, ordering in enumerate(random_orderings):
        ordered_examples = [in_context_data[idx] for idx in ordering]
        accuracy = evaluate_model(llm, test_data, ordered_examples, output_file=output_file)
        results.append(accuracy)
        f.write(f"Ordering {i+1}: Accuracy = {accuracy}\n")
        print(f"Ordering {i+1}: Accuracy = {accuracy}")

for i, accuracy in enumerate(results):
    print(f"Ordering {i+1}: Accuracy = {accuracy}")
