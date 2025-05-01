import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import os

# CONFIG
DATASET_PATHS = {
    "finqa": "data/finqa/processed",
    "bioasq": "data/bioasq/processed",
    "legalbench": "data/legalbench/processed"
}

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Exact Match (EM)
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

# F1 Score Calculation
def compute_f1(pred, truth):
    pred_tokens = pred.lower().split()
    truth_tokens = truth.lower().split()

    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Mean Reciprocal Rank (MRR)
def compute_mrr(predicted_ranks):
    return np.mean([1 / (rank + 1) for rank in predicted_ranks])

# nDCG (Normalized Discounted Cumulative Gain)
def compute_ndcg(retrieved_ranks, relevant_ranks, k=10):
    dcg = 0
    for i in range(min(k, len(retrieved_ranks))):
        if retrieved_ranks[i] in relevant_ranks:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(relevant_ranks)))])
    return dcg / idcg if idcg > 0 else 0

# Function to load datasets from CSV
def load_csv_data(dataset_name):
    train_path = os.path.join(DATASET_PATHS[dataset_name], "train.csv")
    test_path = os.path.join(DATASET_PATHS[dataset_name], "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    else:
        print(f"❌ {dataset_name} - Missing 'train.csv' or 'test.csv'. Skipping...")
        return None, None

# Evaluate baseline for each dataset
def evaluate_baseline(dataset_name):
    print(f"\n🚀 Evaluating baseline for {dataset_name}...\n")
    
    # Load dataset
    train_data, test_data = load_csv_data(dataset_name)
    if train_data is None or test_data is None:
        return

    questions = test_data["question"]
    contexts = test_data["context"]
    answers = test_data["answer"]

    # Detect invalid contexts
    invalid_indices = [
        i for i, c in enumerate(contexts)
        if not c or (isinstance(c, str) and c.strip() == "") or (isinstance(c, list) and len(c) == 0)
    ]
    print(f"⚠️ Found {len(invalid_indices)} empty/missing contexts out of {len(contexts)} samples.")

    # Filter valid entries
    valid_data = [
        (q, c, a)
        for q, c, a in zip(questions, contexts, answers)
        if c and ((isinstance(c, str) and c.strip() != "") or (isinstance(c, list) and len(c) > 0))
    ]

    if len(valid_data) == 0:
        print(f"❌ No valid data to evaluate for {dataset_name}. Skipping...")
        return

    filtered_questions, filtered_contexts, filtered_answers = zip(*valid_data)

    # Normalize contexts
    normalized_contexts = [
        c if isinstance(c, str) else " ".join(c)
        for c in filtered_contexts
    ]

    # Compute embeddings
    model = SentenceTransformer(MODEL_NAME)
    context_embeddings = model.encode(normalized_contexts, convert_to_tensor=True)

    ranks = []
    predicted_answers = []
    true_answers = []
    results = []

    # Store ranks for MRR and nDCG computation
    predicted_ranks = []
    relevant_ranks = []

    for idx, question in tqdm(enumerate(filtered_questions), total=len(filtered_questions)):
        q_emb = model.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, context_embeddings)[0]

        top_rank = np.argmax(cos_scores.cpu().numpy())

        retrieval_correct = 1 if top_rank == idx else 0

        predicted_answer = filtered_answers[top_rank]
        true_answer = filtered_answers[idx]

        em = exact_match(predicted_answer, true_answer)
        f1 = compute_f1(predicted_answer, true_answer)

        # Store ranks for MRR and nDCG
        predicted_ranks.append(top_rank)
        relevant_ranks.append(idx)

        # Store detailed result
        results.append({
            "question": question,
            "ground_truth_context": normalized_contexts[idx],
            "ground_truth_answer": true_answer,
            "predicted_context": normalized_contexts[top_rank],
            "predicted_answer": predicted_answer,
            "retrieval_correct": retrieval_correct,
            "exact_match": em,
            "f1_score": f1
        })

    # After loop, compute additional metrics
    mrr = compute_mrr(predicted_ranks)
    ndcg = compute_ndcg(predicted_ranks, relevant_ranks)

    # After loop, save DataFrame
    df = pd.DataFrame(results)
    csv_path = f"results/{dataset_name}_baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved detailed results to {csv_path}")

    # Print metrics
    print(f"📊 Evaluation Metrics for {dataset_name}:")
    print(f"✅ Exact Match (EM): {np.mean([res['exact_match'] for res in results]):.4f}")
    print(f"✅ F1 Score: {np.mean([res['f1_score'] for res in results]):.4f}")
    print(f"✅ MRR: {mrr:.4f}")
    print(f"✅ nDCG: {ndcg:.4f}")

if __name__ == "__main__":
    for dataset in DATASET_PATHS.keys():
        evaluate_baseline(dataset)
