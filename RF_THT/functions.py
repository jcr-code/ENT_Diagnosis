import numpy as np
import matplotlib.pyplot as plt
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def precision(y_true, y_pred, class_label):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    false_positives = np.sum((y_true != class_label) & (y_pred == class_label))

    if (true_positives + false_positives == 0) :
        return 0
    else:
        return true_positives / (true_positives + false_positives)

def calcFPR(y_true, y_pred, class_label):
    false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
    true_negatives = np.sum((y_true != class_label) & (y_pred != class_label))
    if(false_positives + true_negatives == 0):
        return 0
    else:
        return false_positives / (false_positives + true_negatives)

def recall(y_true, y_pred, class_label):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    false_negatives = np.sum((y_true == class_label) & (y_pred != class_label))
    if(true_positives + false_negatives == 0):
        return 0
    else:
        return true_positives / (true_positives + false_negatives)

def f1_score(y_true, y_pred, class_label):
    prec = precision(y_true, y_pred, class_label)
    rec = recall(y_true, y_pred, class_label)
    if(prec + rec == 0):
        return 0
    else:
        return 2 * (prec * rec) / (prec + rec)

def macro_average_f1_score(y_true, y_pred):   
    f1_scores = [f1_score(y_true, y_pred, class_label) for class_label in np.unique(y_true)]
    return np.mean(f1_scores)

def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)))
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix

def plot_confusion_matrix(matrix, labels):
    print("\nConfusion Matrix:")
    for row in matrix:
        print(row)
    
    print("\nConfusion Matrix (Normalized):")
    total_samples = sum(sum(row) for row in matrix)
    for i, row in enumerate(matrix):
        row_total = sum(row)
        normalized_row = ["{:.2f}".format(val/row_total) if row_total != 0 else "0.00" for val in row]
        print(normalized_row)
    
    print("\nPlotting Confusion Matrix:")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            print(val, end=" ")
        print()

def plot_confusion_matrix_2(matrix, labels):
    fig, ax = plt.subplots()
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues')
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, "{:.0f}".format(matrix[i, j]), ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()
# def roc_curve(y_true, y_pred_probs):
#     num_classes = len(np.unique(y_true))
#     roc_points = []

#     for class_label in range(num_classes):
#         class_tpr = []
#         class_fpr = []
#         y_pred_binary = (y_pred_probs == class_label).astype(int)  # Convert predicted labels to binary
#         tpr = recall(y_true, y_pred_binary, class_label)
#         fpr = calcFPR(y_true, y_pred_binary, class_label)
#         class_tpr.append(tpr)
#         class_fpr.append(fpr)
#         roc_points.append((class_fpr, class_tpr))

#     return roc_points

# def plot_roc_curve(roc_points):
#     plt.figure(figsize=(8, 6))
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')

#     for class_label, (class_fpr, class_tpr) in enumerate(roc_points):
#         plt.plot(class_fpr, class_tpr, label=f'Class {class_label}')

#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.show()