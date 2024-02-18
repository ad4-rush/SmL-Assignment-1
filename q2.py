import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = np.load('mnist.npz')
x_train, y_train = mnist['x_train'], mnist['y_train']

# Choose 100 samples from each class and create a 784×1000 data matrix X
num_samples_per_class = 100
selected_samples = []

for i in range(10):
    class_samples = x_train[y_train == i][:num_samples_per_class]
    selected_samples.append(class_samples.reshape(-1, 28*28))

X = np.hstack(selected_samples)

# Remove mean from X
X_centered = X - np.mean(X, axis=1, keepdims=True)

# Apply PCA on the centralized X
covariance_matrix = np.cov(X_centered)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
U = eigenvectors

# Perform Y = U^T X and reconstruct X_recon = UY
Y = U.T @ X_centered
X_recon = U @ Y

# Check the MSE between X and X_recon
mse = mean_squared_error(X_centered, X_recon)
print(f'Mean Squared Error between X and X_recon: {mse}')

# Plot the original and reconstructed images for the first class
num_eigenvectors_to_plot = [5, 10, 20]

for p in num_eigenvectors_to_plot:
    UpY = U[:, :p] @ Y[:p, :]
    X_recon_p = UpY + np.mean(X, axis=1, keepdims=True)
    
    # Reshape each column to 28x28 and plot the images
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_recon_p[:, i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Reconstructed Images with {p} Eigenvectors')
    plt.show()

# Let X_test be your test set
x_test, y_test = mnist['x_test'], mnist['y_test']
X_test = x_test.reshape(-1, 28*28)

# Find Y = U_p^T X_test for each value of p
accuracy_results = []

for p in num_eigenvectors_to_plot:
    UpY_test = U[:, :p].T @ X_test.T
    UpY_test = UpY_test.T
    
    # Apply QDA from Q1 on Y
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(Y.T[:, :p], y_train)
    y_pred_test = qda.predict(UpY_test)
    
    # Obtain accuracy on the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_results.append(accuracy_test)

    # Obtain per class accuracy
    class_accuracy = np.zeros(10)
    for i in range(10):
        class_accuracy[i] = accuracy_score(y_test[y_test == i], y_pred_test[y_test == i])

    print(f'Accuracy with {p} Eigenvectors on Test Set: {accuracy_test}')
    print('Per Class Accuracy:', class_accuracy)
