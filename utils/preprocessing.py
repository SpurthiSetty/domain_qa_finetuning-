from datasets import load_dataset
import random
import re
from transformers import AutoTokenizer
from datasets import concatenate_datasets


# Function to standardize column names and split into question, context, and answer
def preprocess_bioasq(bioasq_dataset):
    def parse_text(example):
        raw_text = example["text"]
        # Use regex to extract <answer> ... <context> ...
        answer_match = re.search(r"<answer>(.*?)<context>", raw_text, re.DOTALL)
        context_match = re.search(r"<context>(.*)", raw_text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        context = context_match.group(1).strip() if context_match else ""
        return {
            "question": example["question"],
            "context": context,
            "answer": answer
        }
    return bioasq_dataset.map(parse_text)


def preprocess_finqa(finqa_dataset):
    def map_finqa(example):
        return {
            "question": example["question"],
            "context": example["post_text"],  # FinQA has financial tables/posts
            "answer": example["final_result"]
        }
    return finqa_dataset.map(map_finqa)

def preprocess_legalbench(legalbench_dataset, dataset_name):
    def map_legal(example):
        # Check if 'contract' exists in the dataset
        if "contract" in example:
            context = example["contract"]
        else:
            context = example["text"] if "text" in example else example["context"]

        return {
            "question": example["question"],
            "context": context,  # Correctly renamed here
            "answer": example["answer"]
        }

    # Apply the mapping function
    processed_data = legalbench_dataset.map(map_legal)

    # For contract_qa, remove the redundant 'contract' column after processing
    if dataset_name == "consumer_contracts_qa":
        processed_data = processed_data.remove_columns(["contract"])  # Remove 'contract' column for contract_qa

    return processed_data


# Tokenization function
def tokenize_batch(batch):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer(
        text=[str(q) for q in batch["question"]],
        text_pair=[str(c) for c in batch["context"]],
        truncation=True,
        padding="max_length"
    )


# Downsampling function to ensure dataset sizes are comparable
def downsample_dataset(dataset, target_size=5000):
    if len(dataset) > target_size:
        return dataset.select(random.sample(range(len(dataset)), target_size))  # Random downsampling
    return dataset

# Function to split and downsample datasets
def split_and_downsample(dataset, target_train_size=5000, target_test_size=1000):
    # Shuffle the dataset to ensure randomness
    dataset = dataset.shuffle(seed=42)
    
    # Split the dataset into train and test (80/20 split)
    train_size = int(0.8 * len(dataset))
    dataset_train = dataset.select(range(train_size))
    dataset_test = dataset.select(range(train_size, len(dataset)))
    
    # Downsample the train set
    dataset_train = downsample_dataset(dataset_train, target_size=target_train_size)
    
    # Optionally downsample the test set
    dataset_test = downsample_dataset(dataset_test, target_size=target_test_size)
    
    return dataset_train, dataset_test

# Tokenization function
def tokenize_batch(batch):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer(
        text=[str(q) for q in batch["question"]],
        text_pair=[str(c) for c in batch["context"]],
        truncation=True,
        padding="max_length"
    )

# Save function to store tokenized data
def save_to_disk(dataset, path):
    dataset.save_to_disk(path)

# Main function
def main():
    # Loading datasets
    bio_asq = load_dataset("kroshan/BioASQ")
    fin_qa = load_dataset("ibm-research/finqa")
    legal_bench = load_dataset("nguha/legalbench", "contract_qa")
    legal_bench_2 = load_dataset("nguha/legalbench", "consumer_contracts_qa")
    legal_bench_3 = load_dataset("nguha/legalbench", "privacy_policy_qa")

        # Preprocess datasets
    bioasq_processed = preprocess_bioasq(bio_asq)
    finqa_processed = preprocess_finqa(fin_qa)
        # Preprocess for different legal tasks
    legalbench_processed_1 = preprocess_legalbench(legal_bench, "contract_qa")
    legalbench_processed_2 = preprocess_legalbench(legal_bench_2, "consumer_contracts_qa")
    legalbench_processed_3 = preprocess_legalbench(legal_bench_3, "privacy_policy_qa")

    # Apply the function to BioASQ
    bio_asq_combined = concatenate_datasets([bioasq_processed["train"], bioasq_processed["validation"]])
    bioasq_train, bioasq_test = split_and_downsample(bio_asq_combined)

    # Apply the function to FinQA
    finqa_train, finqa_test = split_and_downsample(finqa_processed["train"])

    # Concatenate train and test splits for legalbench datasets
    legalbench_combined_train = concatenate_datasets([legalbench_processed_1["train"], legalbench_processed_2["train"], legalbench_processed_3['train']])
    legalbench_combined_test = concatenate_datasets([legalbench_processed_1["test"], legalbench_processed_2["test"], legalbench_processed_3['test']])

    legalbench_combined = concatenate_datasets([legalbench_combined_train, legalbench_combined_test])

    # Now split the combined dataset into train and test (80/20 split) before downsampling
    legalbench_combined = legalbench_combined.shuffle(seed=42)  # Shuffle to ensure randomness

    # Split the combined dataset into train and test (80/20 split)
    train_size = int(0.8 * len(legalbench_combined))
    legalbench_train = legalbench_combined.select(range(train_size))
    legalbench_test = legalbench_combined.select(range(train_size, len(legalbench_combined)))

    # Downsample the train set to ensure the total size is 5000 rows
    legalbench_train = downsample_dataset(legalbench_train, target_size=5000)

    # Optionally downsample the test set (if you want the test size to also be consistent, but usually, the test set size is kept fixed)
    legalbench_test = downsample_dataset(legalbench_test, target_size=1000)  # Optional for test set

    # Tokenize the datasets
    tokenized_bioasq = bioasq_train.map(tokenize_batch, batched=True)
    tokenized_finqa = finqa_train.map(tokenize_batch, batched=True)
    tokenized_legalbench = legalbench_train.map(tokenize_batch, batched=True)

    # Save the tokenized datasets to disk
    save_to_disk(tokenized_bioasq, "data/bioasq/processed")
    save_to_disk(tokenized_finqa, "data/finqa/processed")
    save_to_disk(tokenized_legalbench, "data/legalbench/processed")

    # Check the lengths to ensure proper split
    print("Legal Bench Train Set Size:", len(legalbench_train))
    print("Legal Bench Test Set Size:", len(legalbench_test))
    print("BioASQ Train Set Size:", len(bioasq_train))
    print("BioASQ Test Set Size:", len(bioasq_test))
    print("FinQA Train Set Size:", len(finqa_train))
    print("FinQA Test Set Size:", len(finqa_test))


    
if __name__ == "__main__":
    main()
