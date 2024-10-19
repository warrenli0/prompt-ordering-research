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
    prompt = ""
    if len(in_context_examples) > 0:
        prompt += "Given are the answers for the following questions as an index from 0 to 3 for the given choices:\n"
    for ex in in_context_examples:
        prompt += f"Q: {ex['question']}\nChoices: {ex['choices']}\nA: {ex['answer']}\n\n"
    prompt += "Answer this question as an index from 0 to 3 for the given answer choices:\n"
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

def create_prompt_custom(example, in_context_examples, label_names):
    prompt = "Classify the text into one of the following categories: " + ", ".join(label_names) + ".\n\n"
    for ex in in_context_examples:
        prompt += f"Text: {ex['text']}\nCategory: {label_names[ex['label']]}\n\n"
    prompt += "Classify this text:\n"
    prompt += f"Text: {example['text']}\nCategory:"
    return prompt

def apply_chat_template(prompt):
    # Format the prompt as a chat-style conversation
    chat_prompt = "<bos><start_of_turn>user\n"
    chat_prompt += prompt
    chat_prompt += "<end_of_turn>\n"
    chat_prompt += "<start_of_turn>model\n"
    return chat_prompt

# Helper function to get the appropriate prompt creator based on dataset name
def get_prompt_creator(dataset_name, is_instruction_tuned=False, label_names=None):
    # Get the original prompt creator function based on dataset name
    if dataset_name == 'gsm8k':
        prompt_creator = create_prompt_gsm8k
    elif dataset_name == 'bbh':
        prompt_creator = create_prompt_bbh
    elif dataset_name == 'mmlu':
        prompt_creator = create_prompt_mmlu
    elif dataset_name == 'ag_news':
        prompt_creator = create_prompt_agnews
    elif dataset_name == 'sst2':
        prompt_creator = create_prompt_sst2
    elif dataset_name == 'dbpedia':
        prompt_creator = create_prompt_dbpedia
    elif dataset_name in ['nyt-topics', 'nyt-locations']:
        prompt_creator = lambda ex, in_context: create_prompt_custom(ex, in_context, label_names)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Return a new function that wraps the original prompt creator
    def wrapped_prompt_creator(example, in_context_examples):
        # Generate the prompt using the original creator
        prompt = prompt_creator(example, in_context_examples)

        if not is_instruction_tuned: return prompt
        # Apply the chat template if needed
        return apply_chat_template(prompt)
    
    return wrapped_prompt_creator if is_instruction_tuned else prompt_creator
