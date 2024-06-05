import random
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
token = "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"

login(token=token)

llm = LLM(model=model_name, gpu_memory_utilization=0.5, enforce_eager=True)

bbh = load_dataset('lukaemon/bbh', 'main')['test'].select(range(100))

def create_prompt(example, in_context_examples):
    prompt = ""
    for ex in in_context_examples:
        prompt += f"Q: {ex['input']}\nA: {ex['target']}\n\n"
    prompt += "Give me the answer for the following question:\n"
    prompt += f"Q: {example['input']}\nA: "
    return prompt

def evaluate_model(llm, dataset, in_context_examples, max_length=256, output_file=None):
    correct = 0
    total = 0

    for example in tqdm(dataset):
        prompt = create_prompt(example, in_context_examples)

        sampling_params = SamplingParams(max_tokens=max_length)

        outputs = llm.generate([prompt], sampling_params=sampling_params)

        generated_text = outputs[0].outputs[0].text
        correct_answer = example["target"].strip()

        is_correct = correct_answer in generated_text

        total += 1
        if is_correct:
            correct += 1

        if output_file:
            output_file.write(f"Q: {example['input']}\n")
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

in_context_data = bbh.select(range(40))
test_data = bbh.select(range(50, 100))

random_orderings = generate_random_orderings(in_context_data, num_orderings=10)

results = []
with open("bbh_evaluation_results.txt", "w") as f, open("bbh_outputs.txt", "w") as output_file:
    for i, ordering in enumerate(random_orderings):
        ordered_examples = [in_context_data[idx] for idx in ordering]
        accuracy = evaluate_model(llm, test_data, ordered_examples, output_file=output_file)
        results.append(accuracy)
        f.write(f"Ordering {i+1}: Accuracy = {accuracy}\n")
        print(f"Ordering {i+1}: Accuracy = {accuracy}")

for i, accuracy in enumerate(results):
    print(f"Ordering {i+1}: Accuracy = {accuracy}")
