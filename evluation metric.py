def evaluate_retrieval(relevant_docs, retrieved_docs):
    tp = len(relevant_docs & retrieved_docs)
    fp = len(retrieved_docs - relevant_docs)
    fn = len(relevant_docs - retrieved_docs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, precision, recall, f1

relevant_documents = {1, 2, 3, 5, 7}
retrieved_documents = {1, 2, 4, 5, 6}

tp, fp, fn, precision, recall, f1_score = evaluate_retrieval(relevant_documents, retrieved_documents)

print("\n=== IR Evaluation Metrics ===")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")