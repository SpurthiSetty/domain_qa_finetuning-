from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# CONFIG
DATASET_PATHS = {
    "finqa": "data/finqa/processed",
    "bioasq": "data/bioasq/processed",
    "legalbench": "data/legalbench/processed"
}

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

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

def evaluate_baseline(dataset_name):
    print(f"\n🚀 Evaluating baseline for {dataset_name}...\n")
    
    # Load dataset and model
    dataset = load_from_disk(DATASET_PATHS[dataset_name])
    model = SentenceTransformer(MODEL_NAME)

    eval_data = dataset["test"] if "test" in dataset else dataset["validation"]
    eval_data = eval_data.select(range(min(100, len(eval_data))))  # Limit to 100 samples for speed

    questions = eval_data["question"]
    contexts = eval_data["context"]
    answers = eval_data["answer"]

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
    context_embeddings = model.encode(normalized_contexts, convert_to_tensor=True)

    ranks = []
    predicted_answers = []
    true_answers = []
    results = []

    for idx, question in tqdm(enumerate(filtered_questions), total=len(filtered_questions)):
        q_emb = model.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, context_embeddings)[0]

        top_rank = np.argmax(cos_scores.cpu().numpy())

        retrieval_correct = 1 if top_rank == idx else 0

        predicted_answer = filtered_answers[top_rank]
        true_answer = filtered_answers[idx]

        em = exact_match(predicted_answer, true_answer)
        f1 = compute_f1(predicted_answer, true_answer)

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

    # After loop, save DataFrame
    df = pd.DataFrame(results)
    csv_path = f"results/{dataset_name}_baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved detailed results to {csv_path}")

    # # Calculate metrics
    # top1_acc = np.mean(ranks)
    # print(f"\n✅ {dataset_name} - Top-1 Retrieval Accuracy: {top1_acc:.4f}")
    
    # em_scores = [exact_match(p, t) for p, t in zip(predicted_answers, true_answers)]
    # answer_em = np.mean(em_scores)
    # print(f"✅ {dataset_name} - Answer Exact Match (EM): {answer_em:.4f}")

    # f1_scores = [compute_f1(p, t) for p, t in zip(predicted_answers, true_answers)]
    # avg_f1 = np.mean(f1_scores)
    # print(f"✅ {dataset_name} - Answer F1 Score: {avg_f1:.4f}")

if __name__ == "__main__":
    for dataset in DATASET_PATHS.keys():
        evaluate_baseline(dataset)
