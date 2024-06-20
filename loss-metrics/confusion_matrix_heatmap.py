import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_confusion_matrix(y_true, y_pred, num_classes):
    y_true = tf.constant(y_true, dtype=tf.int32)
    y_pred = tf.constant(y_pred, dtype=tf.int32)
    
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return confusion_matrix.numpy()

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    num_classes = 2
    class_names = ['Class 0', 'Class 1']
    
    confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    plot_confusion_matrix(confusion_matrix, class_names)
