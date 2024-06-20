import tensorflow as tf

def compute_f1_score(y_true, y_pred):
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    
    precision_metric.update_state(y_true, y_pred)
    recall_metric.update_state(y_true, y_pred)
    
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# Example usage
if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    f1_score = compute_f1_score(y_true, y_pred)
    print(f"F1 Score: {f1_score}")
