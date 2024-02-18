import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Compute mean vectors and covariance matrices for each class
# Load MNIST dataset
test_size= 1000
mnist = np.load('mnist.npz')
x_train, y_train = mnist['x_train'], mnist['y_train']
x_test, y_test = mnist['x_test'][:test_size], mnist['y_test'][:test_size]
# Visualize 5 samples from each class
classes_to_visualize = np.unique(y_train)[:10]
samples_per_class = 5

fig, axs = plt.subplots(len(classes_to_visualize), samples_per_class, figsize=(10, 10))

for i, class_label in enumerate(classes_to_visualize):
    class_samples = x_train[y_train == class_label][:samples_per_class]
    
    for j in range(samples_per_class):
        axs[i, j].imshow(class_samples[j], cmap='gray')
        axs[i, j].axis('off')

plt.show()
# plt.show(block=False)
# plt.pause(5)  # Display the plot for 5 seconds

# # Close the plot
# plt.close()
# Vectorize the images in the test set

x_train_vectorized = x_train.reshape((x_train.shape[0], -1))
x_test_vectorized = x_test.reshape((x_test.shape[0], -1))
mean_vectors = []
cov_matrices = []
log_dets = []
for i in range(10):
    class_samples = x_train[y_train == i].reshape((-1, 784))
    
    # Add regularization to the covariance matrix
    cov_matrix = np.cov(class_samples, rowvar=False)
    cov_matrix_reg = cov_matrix + 1e-3 * np.eye(cov_matrix.shape[0])

    mean_vectors.append(np.mean(class_samples, axis=0))
    cov_matrices.append(cov_matrix_reg)
    
    # Calculate log determinant using cov_matrix_reg
    log_det = np.linalg.slogdet(cov_matrix_reg)[1]
    log_dets.append(log_det)
print(log_dets)
a=0
# QDA prediction function
def qda_predict(sample):
    predictions = []
    for i in range(10):
        global a 
        mean_vector = mean_vectors[i]
        cov_matrix = cov_matrices[i]
        log_det = log_dets[i]
        # QDA discriminant function with log determinant
        log_likelihood = -0.5 * log_det \
                         - 0.5 * np.dot(np.dot((sample - mean_vector), np.linalg.inv(cov_matrix)), (sample - mean_vector).T)
        predictions.append(log_likelihood)
    a+=1
    print(np.argmax(predictions),"                  ",a)
    return np.argmax(predictions)

# Predict the class of samples in the test set using manual QDA
y_pred_manual = np.apply_along_axis(qda_predict, axis=1, arr=x_test_vectorized)

# Calculate class-wise accuracy manually
class_accuracy_manual = np.zeros(10)
for i in range(10):
    correct_class = np.sum((y_test == i) & (y_pred_manual == i))
    total_class_samples = np.sum(y_test == i)
    class_accuracy_manual[i] = correct_class / total_class_samples
    print(f"Class {i} Accuracy:", class_accuracy_manual[i])

# Fit scikit-learn's QDA model
qda_sklearn = QuadraticDiscriminantAnalysis()
qda_sklearn.fit(x_train_vectorized, y_train)

# Predictions from scikit-learn's QDA
y_pred_sklearn = qda_sklearn.predict(x_test_vectorized)

# Print confusion matrices
conf_matrix_manual = confusion_matrix(y_test, y_pred_manual)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)

print("Manual QDA Confusion Matrix:")
print(conf_matrix_manual)
print("Scikit-learn QDA Confusion Matrix:")
print(conf_matrix_sklearn)

# Plot side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Manual QDA confusion matrix
axes[0].imshow(conf_matrix_manual, cmap='Blues', interpolation='nearest')
axes[0].set_title('Manual QDA Confusion Matrix')

# scikit-learn's QDA confusion matrix
axes[1].imshow(conf_matrix_sklearn, cmap='Blues', interpolation='nearest')
axes[1].set_title('Scikit-learn QDA Confusion Matrix')

# Add labels, ticks, etc.
for ax in axes:
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
plt.savefig('Confusion Matrix.png')
plt.show()
# plt.show(block=False)
# plt.pause(15)  # Display the plot for 5 seconds

# # Close the plot
# plt.close()

# Compare accuracies
correct_sklearn = np.sum(y_test == y_pred_sklearn)
accuracy_sklearn = correct_sklearn / len(y_test)
correct_manual = np.sum(y_test == y_pred_manual)
accuracy_manual = correct_manual / len(y_test)
print("Manual QDA Accuracy:", accuracy_manual)
print("Scikit-learn QDA Accuracy:", accuracy_sklearn)