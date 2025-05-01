from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score
import random
from datasets import load_dataset
import pandas as pd 
from baseline import exact_match, compute_f1
# Load pre-trained model for Contrastive Learning (e.g., SBERT)
model = SentenceTransformer('all-mpnet-base-v2')

# Load datasets (you can modify the paths if needed)
DATASET_PATHS = {
    "finqa": "data/finqa/processed",
    "bioasq": "data/bioasq/processed",
    "legalbench": "data/legalbench/processed"
}

# Loading train and test data (Make sure the data is loaded correctly as DataFrames)
train_data = pd.read_csv("data/bioasq/processed/train.csv")  # Example path
test_data = pd.read_csv("data/bioasq/processed/test.csv")  # Example path

# Create positive pairs for training (question, context)
train_samples = []
for i in range(len(train_data)):
    train_samples.append(InputExample(texts=[train_data["question"][i], train_data["context"][i]]))  # Positive pair

# Negative pairs: Randomly sample a different context for the same question
negative_samples = []  # Populate with negative pairs
for i in range(len(train_data)):
    negative_samples.append(InputExample(texts=[train_data["question"][i], random.choice(train_data["context"])]))  # Negative pair

# Combine positive and negative examples
train_samples.extend(negative_samples)

# DataLoader for batching
train_dataloader = DataLoader(train_samples, batch_size=32)

# Define a contrastive loss
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the fine-tuned model
model.save("finetuned_model")

# **Evaluation**
def evaluate_contrastive_model(model, test_data):
    questions = test_data["question"]
    contexts = test_data["context"]
    answers = test_data["answer"]

    # Compute embeddings for the context and questions
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    question_embeddings = model.encode(questions, convert_to_tensor=True)

    ranks = []
    predicted_answers = []
    true_answers = []
    results = []

    # Perform the retrieval-based prediction
    for idx, question in enumerate(questions):
        q_emb = question_embeddings[idx]
        cos_scores = util.cos_sim(q_emb, context_embeddings)[0]

        # Rank the contexts by cosine similarity
        top_rank = torch.argmax(cos_scores).item()

        retrieval_correct = 1 if top_rank == idx else 0  # Check if correct context is retrieved

        predicted_answer = answers[top_rank]
        true_answer = answers[idx]

        # Calculate Exact Match and F1-score
        em = exact_match(predicted_answer, true_answer)
        f1 = compute_f1(predicted_answer, true_answer)

        results.append({
            "question": question,
            "ground_truth_context": contexts[idx],
            "ground_truth_answer": true_answer,
            "predicted_context": contexts[top_rank],
            "predicted_answer": predicted_answer,
            "retrieval_correct": retrieval_correct,
            "exact_match": em,
            "f1_score": f1
        })

    # Save evaluation results to a CSV file
    df = pd.DataFrame(results)
    csv_path = f"results/{dataset_name}_contrastive_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved detailed results to {csv_path}")

    return df

# Evaluate on the test dataset after fine-tuning
evaluate_contrastive_model(model, test_data)
