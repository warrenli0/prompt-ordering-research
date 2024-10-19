import re
import pandas as pd
from vllm import SamplingParams
import inspect
import os

def evaluate_model(llm, dataset_name, dataset, in_context_examples, create_prompt_fn, label_names=None, max_length=256, output_file=None, results_path=None):
    correct = 0
    total = 0
    correctness = []
    predicted_labels = []
    results_list = []

     # Check the number of arguments that create_prompt_fn expects
    num_args = len(inspect.signature(create_prompt_fn).parameters)

    if num_args == 2:
        # For non-custom datasets, only pass example and in_context_examples
        prompts = [create_prompt_fn(example, in_context_examples) for example in dataset]
    elif num_args == 3:
        # For custom datasets, pass example, in_context_examples, and label_names
        prompts = [create_prompt_fn(example, in_context_examples, label_names) for example in dataset]
    else:
        raise ValueError(f"Unexpected number of arguments for create_prompt_fn: {num_args}")

    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, example in enumerate(dataset):
        generated_text = outputs[idx].outputs[0].text

        if dataset_name == 'bbh':
            correct_answer = example["target"].strip()
        elif dataset_name == 'mmlu':
            inv = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            correct_answer = str(example["answer"])
            # generated_text = str(inv.get(generated_text.strip(), -1))
        elif dataset_name == 'ag_news':
            inv = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3}
            correct_answer = str(example["label"])
            generated_text = str(inv.get(generated_text.strip(), -1))
        elif dataset_name == 'sst2':
            inv = {'negative': 0, 'positive': 1}
            correct_answer = str(example["label"])
            generated_text = str(inv.get(generated_text.strip(), -1))
        elif dataset_name == 'dbpedia':
            inv = {'Company': 0, 'School': 1, 'Artist': 2, 'Athlete': 3, 'Politician': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
            # inv = {
            #     'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 
            #     'MeanOfTransportation': 5, 'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal:': 9, 
            #     'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13
            # }
            correct_answer = str(example["label"])
            generated_text = str(inv.get(generated_text.strip(), -1))
        elif dataset_name in ['nyt-topics', 'nyt-locations']:
            inv = {label: idx for idx, label in enumerate(label_names)}
            correct_answer = str(example["label"])
            generated_text = str(inv.get(generated_text.strip(), -1))
        else:
            correct_answer = example["answer"].split('####')[-1].strip()

        is_correct = correct_answer in generated_text
        correctness.append(is_correct)
        predicted_labels.append(generated_text)

        total += 1
        if is_correct:
            correct += 1
        
        # Save results for future reference
        result = {
            'dataset_name': dataset_name,
            'test_example_id': idx,
            'ground_truth_label': correct_answer,
            'predicted_label': generated_text,
            'correctness': is_correct,
        }
        results_list.append(result)

        if output_file and idx < 5:
            if idx == 0: output_file.write(f"Prompt: {prompts[idx]}\n")
            output_file.write(f"Q: {example.get('text', example.get('input', ''))}\n")
            output_file.write(f"Generated A:{generated_text}\n")
            output_file.write(f"Correct A: {correct_answer}\n")
            output_file.write(f"Is Correct: {is_correct}\n")
            output_file.write("\n")
    
    if results_path:
        results_df = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)

    accuracy = correct / total
    return accuracy, correctness, predicted_labels

def evaluate_it_model(llm, dataset_name, dataset, in_context_examples, create_prompt_fn, label_names=None, max_length=256, output_file=None, results_path=None):
    correct = 0
    total = 0
    correctness = []
    predicted_labels = []
    results_list = []

    # Check the number of arguments that create_prompt_fn expects
    num_args = len(inspect.signature(create_prompt_fn).parameters)

    if num_args == 2:
        # For non-custom datasets, only pass example and in_context_examples
        prompts = [create_prompt_fn(example, in_context_examples) for example in dataset]
    elif num_args == 3:
        # For custom datasets, pass example, in_context_examples, and label_names
        prompts = [create_prompt_fn(example, in_context_examples, label_names) for example in dataset]
    else:
        raise ValueError(f"Unexpected number of arguments for create_prompt_fn: {num_args}")

    # Generation settings
    sampling_params = SamplingParams(max_tokens=50, temperature=0)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, example in enumerate(dataset):
        generated_text = outputs[idx].outputs[0].text.strip().lower()  # Normalize generated text

        # Initialize correct answer in text form
        correct_answer_text = None

        label_map = None

        # Map numeric labels to corresponding text labels
        if dataset_name == 'bbh':
            correct_answer_text = example["target"].strip().lower()
        elif dataset_name == 'mmlu':
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            correct_answer_text = label_map.get(example["answer"], '').strip().lower()
        elif dataset_name == 'ag_news':
            label_map = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
            correct_answer_text = label_map.get(example["label"], '').lower()
        elif dataset_name == 'sst2':
            label_map = {0: 'negative', 1: 'positive'}
            correct_answer_text = label_map.get(example["label"], '').lower()
        elif dataset_name == 'dbpedia':
            label_map = {
                0: 'company', 1: 'school', 2: 'artist', 3: 'athlete', 4: 'politician', 
                5: 'transportation', 6: 'building', 7: 'nature', 8: 'village', 9: 'animal', 
                10: 'plant', 11: 'album', 12: 'film', 13: 'book'
            }
            correct_answer_text = label_map.get(example["label"], '').lower()
        elif dataset_name in ['nyt-topics', 'nyt-locations']:
            label_map = {idx: label.lower() for idx, label in enumerate(label_names)}
            correct_answer_text = label_map.get(example["label"], '').lower()
        else:
            correct_answer_text = example["answer"].split('####')[-1].strip().lower()

        # Extract predicted label by matching generated text to label names
        predicted_label = None
        if label_map:
            # Try to find which label is mentioned in the generated text
            for label_idx, label_name in label_map.items():
                if label_name in generated_text:
                    predicted_label = label_idx
                    break
            # If no label is matched, set to -1 or handle accordingly
            if predicted_label is None:
                predicted_label = -1  # Indicates unmatched label
        else:
            # For datasets without label_map, use generated_text as is
            predicted_label = generated_text

        # Check if the correct answer text is present in the generated text
        is_correct = correct_answer_text.lower() in generated_text
        correctness.append(is_correct)
        predicted_labels.append(predicted_label)

        total += 1
        if is_correct:
            correct += 1
        
        result = {
            'dataset_name': dataset_name,
            'test_example_id': idx,
            'ground_truth_label': correct_answer_text,
            'predicted_label': generated_text,
            'correctness': is_correct,
        }
        results_list.append(result)

        # Write output for debugging if output_file is provided
        if output_file and idx < 5:
            if idx == 0:
                output_file.write(f"Prompt: {prompts[idx]}\n")
            output_file.write(f"Q: {example.get('text', example.get('input', ''))}\n")
            output_file.write(f"Generated A: {generated_text}\n")
            output_file.write(f"Correct A (text): {correct_answer_text}\n")
            output_file.write(f"Is Correct: {is_correct}\n")
            output_file.write("\n")
    
    if results_path:
        results_df = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)

    accuracy = correct / total
    return accuracy, correctness, predicted_labels