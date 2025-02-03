import re
import pandas as pd
from vllm import SamplingParams
import inspect
import os
import math
import random
import inspect

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

    max_tokens = 128 if dataset_name == 'gsm8k' else 1
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
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
        elif dataset_name == 'gsm8k':
            # print("Ans:", generated_text)
            correct_answer = example["answer"].split('####')[-1].strip()
            if '####' in generated_text:
                # Split the text by '####', take the first word after the first occurrence
                parts = generated_text.split('####', 1)[-1].split(maxsplit=1)
                generated_text = parts[0].strip() if parts else generated_text  # Check if the split has content
            # print("Generated ans:", generated_text)

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
    # print("Accuracy:", accuracy)
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


def get_label_distribution_for_example(llm,
                                       example,
                                       in_context_examples,
                                       create_prompt_fn,
                                       label_names,
                                       temperature=1.0,
                                       block_ngram_repeat=2,
                                       max_tokens=1):
    """
    Returns a dict: {label_value: probability} for each label in label_names,
    by appending each label to the prompt and extracting log-probs using vLLM.
    """
    # Build the "base prompt" (without the label appended)
    num_args = len(inspect.signature(create_prompt_fn).parameters)
    if num_args == 2:
        base_prompt = create_prompt_fn(example, in_context_examples)
    else:
        base_prompt = create_prompt_fn(example, in_context_examples, label_names)

    # Initialize a dictionary to store label probabilities
    label_logprobs = {}

    # Request log-probs for the first token of each label
    for label in label_names:
        # Construct the full prompt by appending the label to the base prompt
        full_prompt = base_prompt + " " + label

        # Set sampling parameters with log-probs enabled
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1  # Request log-probs for the generated token
        )

        # Generate output with vLLM
        outputs = llm.generate([full_prompt], sampling_params=sampling_params)

         # Extract the list of log-probs dictionaries for the generated sequence
        logprobs_list = outputs[0].outputs[0].logprobs

        if not logprobs_list or not isinstance(logprobs_list, list):
            raise ValueError(f"Unexpected log-probs format for label '{label}'.")

        # Look for the top-ranked token's log-prob (rank=1) in the first dictionary
        top_token_logprob = None
        for token_probs in logprobs_list:
            for token_id, logprob_obj in token_probs.items():
                if logprob_obj.rank == 1:  # Look for the top-ranked token
                    top_token_logprob = logprob_obj.logprob
                    break
            if top_token_logprob is not None:
                break  # Exit after finding the top-ranked token

        if top_token_logprob is None:
            raise ValueError(f"No top-ranked token found for label '{label}'.")

        label_logprobs[label] = top_token_logprob

    # Normalize log-probs into probabilities
    max_logprob = max(label_logprobs.values())  # Avoid overflow
    unnormalized_probs = {label: math.exp(logprob - max_logprob) for label, logprob in label_logprobs.items()}
    normalization_constant = sum(unnormalized_probs.values())
    normalized_probs = {label: prob / normalization_constant for label, prob in unnormalized_probs.items()}

    return normalized_probs


def evaluate_permutation_for_entropy(llm,
                                     in_context_examples,
                                     probing_set,
                                     create_prompt_fn,
                                     label_names,
                                     temperature=2.0,
                                     max_tokens=128,
                                     block_ngram_repeat=2,
                                     metric="localE"):
    """
    Compute GlobalE or LocalE for the given permutation (in_context_examples),
    following the paper:
      GlobalE: 
        1) For each example, pick the top label (argmax).
        2) Count how many times each label is chosen => label distribution
        3) Entropy of that distribution => GlobalE
      LocalE:
        1) For each example, get the *full* distribution over labels p^v_{i,m}
        2) Compute the entropy of that distribution
        3) Average those entropies => LocalE

    :param metric: "globalE" or "localE"
    :return: A float score, higher = "more entropic"
    """
    if not label_names:
        # If your dataset has a known label set, you must pass it here
        raise ValueError("Need a label_names list to compute label distributions.")

    # We'll store either per-example top labels (for globalE)
    # or full distributions (for localE).
    all_label_distributions = []
    all_top_labels = []

    for example in probing_set:
        # 1) Get p^v for each label v
        p_v = get_label_distribution_for_example(
            llm,
            example,
            in_context_examples,
            create_prompt_fn,
            label_names,
            temperature=temperature,
            block_ngram_repeat=block_ngram_repeat,
            max_tokens=5
        )
        all_label_distributions.append(p_v)

        # 2) For global E, we need the top label only
        top_label = max(p_v.items(), key=lambda x: x[1])[0]  # label with highest prob
        all_top_labels.append(top_label)

    if metric.lower() == "globale":
        # GlobalE => entropy of the distribution over the entire dataset
        # 1) Count how many times each label was chosen
        label_counts = {lbl: 0 for lbl in label_names}
        for lbl in all_top_labels:
            label_counts[lbl] += 1

        # 2) Convert to probabilities
        total_samples = len(all_top_labels)
        label_probs = [count / total_samples for count in label_counts.values()]

        # 3) Entropy = sum( -p log p )
        globale = 0.0
        for p in label_probs:
            if p > 0:
                globale += -p * math.log(p, 2) # base 2
        return globale

    else:  # "localE"
        # LocalE => average of per-example entropies
        # For each p_v for example i, compute H_i = sum_v -p^v log p^v
        # Then average across i
        total_entropy = 0.0
        for p_v in all_label_distributions:
            h_i = 0.0
            for prob in p_v.values():
                if prob > 0:
                    h_i += -prob * math.log(prob, 2)
            total_entropy += h_i
        localE = total_entropy / len(all_label_distributions) if all_label_distributions else 0
        return localE