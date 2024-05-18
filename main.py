import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

gsm8k = load_dataset('gsm8k')

def evaluate_model(model, tokenizer, dataset, prompt_order):
    # Implement evaluation logic here
    return random.random()

def generate_random_orderings(data, num_orderings=10):
    orderings = []
    for _ in range(num_orderings):
        order = list(range(len(data)))
        random.shuffle(order)
        orderings.append(order)
    return orderings

data = list(range(50))
random_orderings = generate_random_orderings(data)

# Evaluate across different orderings
results = []
for ordering in random_orderings:
    result = evaluate_model(model, tokenizer, gsm8k, ordering)
    results.append(result)

print(results)
