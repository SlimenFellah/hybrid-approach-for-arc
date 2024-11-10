import numpy as np

def accuracy_score(true_output, predicted_output):
    """Calculates the accuracy by comparing the true output and predicted output."""
    total_pixels = np.prod(true_output.shape)
    correct_pixels = np.sum(true_output == predicted_output)
    return correct_pixels / total_pixels

def top_k_error_rate(true_outputs, predicted_outputs, k=3):
    """
    Calculates the top-k error rate for ARC tasks.
    For a given true output, checks if it matches any of the k predicted outputs.
    """
    errors = 0
    for true, predictions in zip(true_outputs, predicted_outputs):
        if not any(np.array_equal(true, pred) for pred in predictions[:k]):
            errors += 1
    return errors / len(true_outputs)
