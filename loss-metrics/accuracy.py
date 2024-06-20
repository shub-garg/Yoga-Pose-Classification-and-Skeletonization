import tensorflow as tf

def compute_accuracy(y_true, y_pred):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(y_true, y_pred)
    return metric.result().numpy()

# Example usage
if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    accuracy = compute_accuracy(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
