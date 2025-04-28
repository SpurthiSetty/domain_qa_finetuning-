from datasets import load_dataset
import re
from transformers import AutoTokenizer



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
            "context": example["post_text"],   # FinQA has financial tables/posts
            "answer": example["final_result"]
        }
    return finqa_dataset.map(map_finqa)

def preprocess_legalbench(legalbench_dataset):
    def map_legal(example):
        return {
            "question": example["question"],
            "context": example["text"],        # Legal documents or clauses
            "answer": example["answer"]
        }
    return legalbench_dataset.map(map_legal)


def tokenize_batch(batch):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer(
        text=[str(q) for q in batch["question"]],
        text_pair=[str(c) for c in batch["context"]],
        truncation=True,
        padding="max_length"
    )

def main():

    # Loading datasets
    bio_asq = load_dataset("kroshan/BioASQ")

    fin_qa = load_dataset("ibm-research/finqa")

    legal_bench = load_dataset("nguha/legalbench", "contract_qa")

    bioasq_processed = preprocess_bioasq(bio_asq)
    finqa_processed = preprocess_finqa(fin_qa)
    legalbench_processed = preprocess_legalbench(legal_bench)

    # Tokenize
    tokenized_finqa = finqa_processed.map(tokenize_batch, batched=True)
    tokenized_bioasq = bioasq_processed.map(tokenize_batch, batched=True)
    tokenized_legalbench = legalbench_processed.map(tokenize_batch, batched=True)

    # Save
    tokenized_finqa.save_to_disk("data/finqa/processed")
    tokenized_bioasq.save_to_disk("data/bioasq/processed")
    tokenized_legalbench.save_to_disk("data/legalbench/processed")

if __name__ == "__main__":
    main()
