import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#Path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# ******
# DATA
# ******
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


## ********************************************************
## PREPROCESSING QUESTIONS
## ********************************************************
def preprocessing_Q1():
    """
    Split X and y into training and test sets using an 80/20 split with stratify=y 
    and random_state=42. Print the shapes of all four arrays.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
    return X_train, X_test, y_train, y_test

def preprocessing_Q2(X_train, X_test):
    """
    Fit a StandardScaler on X_train and use it to transform both X_train and X_test. 
    Print the mean of each column in X_train_scaled -- they should all be very close to 0. 
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
          
    # the scaler was fit on X_train only to avoid data leakage. This ensures that the scaling parameters 
    # (mean and standard deviation) are computed only from the training data, which simulates a real-world 
    # scenario where we don't have access to future data when building our model.
    
    return X_train_scaled, X_test_scaled

## ********************************************************
## KNN QUESTIONS
## ********************************************************

def kneighbors_classifier(X_train, X_test, y_train, y_test):
    """
    Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), 
    and predict on the test set. 
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), y_pred
     

def kneighbors_classifier_scaled(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). 
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Scaling slightly hurt performance on this particular split (1.0 to 0.93), but this is likely due to chance in the test split. 
    # Iris features are already on similar scales, so scaling generally makes little difference for this dataset.
    
    return accuracy

def cross_val_scores(X_train, y_train):
    """
    Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. 
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_train, y_train, cv=5)

    # Cross-validation is more trustworthy than a single train/test split because it evaluates the model on 5 different 
    # subsets of the data, reducing the risk of a lucky or unlucky split affecting the result.

    return scores

def KNN_Q4(X_train, y_train):
    """
    Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. 
    For each, compute 5-fold cross-validation accuracy on the unscaled training 
    data and print k and the mean CV score. 
    """
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        print(f"k: {k}, Mean CV Score: {np.mean(scores)}")

    # I choose k=5 because it provides a good balance between bias and variance. 
    # A smaller k (like k=1) can lead to a model that is too sensitive to noise in the training data (high variance), while a larger k (like k=12) can lead 
    # to a model that is too simple and may not capture the underlying patterns in the data (high bias). 


# **************************************************************************
# CLASSIFIER, SKLEARN(DECISION TREE) AND LOGISTIC REGRESSION QUESTIONS
# **************************************************************************
# Classifier Evaluation
def confussion_matrix(y_test, y_pred):
    """
    Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, 
    passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png. 
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.savefig(os.path.join(OUTPUT_DIR, "knn_confusion_matrix.png"))

    # The model makes no errors on this test set -- all off-diagonal cells are 0.
    # No pair of species is confused. This is likely due to the clean nature of the Iris dataset
    # and the favorable random_state=42 split.

# sklearn API: Decision Trees
def decision_tree_classifier(X_train, X_test, y_train, y_test):
    """
    Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, 
    and predict on the test set. Print the accuracy score and classification report. 
    """
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Decision Tree accuracy (0.97) is slightly lower than KNN (1.0) on this split,
    # but both are excellent. The difference is small and likely due to the favorable
    # split for KNN rather than a real performance gap.

    # Scaling would not affect a Decision Tree because it splits on thresholds applied
    # to one feature at a time -- the scale of the values does not change where the
    # best split is found.
    
    return accuracy, report

# Logistic Regression
def logistic_regression(X_train_scaled, y_train):
    """
    Train three logistic regression models on the scaled Iris data, identical in every way except 
    for the C parameter: C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three. 
    For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). 
    """
    C_values = [0.01, 1.0, 100]
    for C in C_values:
        # NOTE: The exercise specifies solver='liblinear', but liblinear does not support
        # multiclass classification (n_classes >= 3). Iris has 3 classes, so it raises:
        # ValueError: The 'liblinear' solver does not support multiclass classification.
        # Changing to solver='lbfgs', which is sklearn's default and handles multiclass natively.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        total_coef_magnitude = np.abs(model.coef_).sum()
        print(f"C: {C}, Total Coefficient Magnitude: {total_coef_magnitude}")

    # As C increases, total coefficient magnitude grows. This is because C is the inverse of regularization strength 
    # - a small C applies heavy regularization, forcing coefficients toward zero to prevent over-reliance on any single feature.
    # A large C applies almost no regularization, allowing coefficients to grow freely.

#*************************************************
# PCA
#*************************************************
def pca_Q1():
    """
    Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), 
    using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png. 
    (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, 
    so digits appear as dark ink on a light background, which is more readable than the default.)
    """    
    plt.figure(figsize=(10, 2))
    for digit in range(10):
        plt.subplot(1, 10, digit + 1)
        plt.imshow(images[digit], cmap='gray_r')
        plt.title(f"Digit: {y_digits[digit]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_digits.png'))
    plt.show()

def pca_Q2():
    """
    Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). 
    As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] is the weighting 
    for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.
    Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar. 
    Here is the pattern for coloring by a label array and attaching a colorbar:
    scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
    plt.colorbar(scatter, label='Digit')
    Save the figure to outputs/pca_2d_projection.png.
    """
    pca = PCA()
    pca.fit(X_digits)
    scores = pca.transform(X_digits)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
    plt.colorbar(scatter, label='Digit')
    plt.title('PCA Projection of Digits Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_2d_projection.png'))
    plt.show()

    # Yes, same-digit images tend to cluster together in this 2D space, although with some overlap between certain digits. 
    # This suggests that the first two principal components capture meaningful structure that separates digit classes.

    return scores, pca
    
def pca_Q3(pca):
    """
    Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components 
    using np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png. 
    """
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(0.8, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_variance_explained.png'))
    plt.show()

    # Approximately 12 components are needed to explain 80% of the variance.

def reconstruct_digit(sample_idx, scores, pca, n_components):
        """Reconstruct one digit using the first n_components principal components."""
        reconstruction = pca.mean_.copy()
        for i in range(n_components):
            reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
        return reconstruction.reshape(8, 8)

def pca_Q4(scores, pca):
    """
    The preprocessing lesson showed that a reconstruction is built by starting from the mean and adding each component weighted by its score. 
    Here is the same idea generalized to n components     
    Using reconstruct_digit(sample_idx, scores, pca, n_components) function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits 
    using reconstruction through principal components n = 2, 5, 15, and 40.
    Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. Add an "Original" row at 
    the top (use images[i], which is already shaped as (8, 8)). Save to outputs/pca_reconstructions.png.
    """
    n_values = [2, 5, 15, 40]
    plt.figure(figsize=(15, 8))
    
    # Original images row
    for i in range(5):
        plt.subplot(len(n_values) + 1, 5, i + 1)
        plt.imshow(images[i], cmap='gray_r')
        plt.title(f"Original: {y_digits[i]}")
        plt.axis('off')
    
    # Reconstructed images rows
    for row_idx, n in enumerate(n_values):
        for col_idx in range(5):
            sample_idx = col_idx  # reconstruct the first 5 digits
            reconstruction = reconstruct_digit(sample_idx, scores, pca, n)
            plt.subplot(len(n_values) + 1, 5, (row_idx + 1) * 5 + col_idx + 1)
            plt.imshow(reconstruction, cmap='gray_r')
            plt.title(f"n={n}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_reconstructions.png'))
    plt.show()

    # Digits become recognizable at around n=5, and improve noticeably at n=15.
    # This matches the variance curve, which captures most of its gain before component 15 
    # - after that the curve flattens and additional components add little visible improvement.
    

   

# PQ1
print("Preprocessing Question 1:")
x_train, x_test, y_train, y_test = preprocessing_Q1()
print(f"x Train set size: {x_train.shape}")
print(f"x Test set size: {x_test.shape}")
print(f"y Train set size: {y_train.shape}")
print(f"y Test set size: {y_test.shape}")

# PQ2
print("\nPreprocessing Question 2:")
x_train_scaled, x_test_scaled = preprocessing_Q2(x_train, x_test)
print("Mean of each column in X_train_scaled:")
print(x_train_scaled.mean(axis=0))

# KNNQ1
print("\nKNN Question 1:")
accuracy, report, y_pred = kneighbors_classifier(x_train, x_test, y_train, y_test)
print("KNN Classifier Accuracy:", accuracy)
print("KNN Classification Report:\n", report)

# KNNQ2
print("\nKNN Question 2:")
accuracy_scaled = kneighbors_classifier_scaled(x_train_scaled, x_test_scaled, y_train, y_test)
print("KNN Classifier Accuracy with Scaled Data:", accuracy_scaled)

# KNNQ3
print("\nKNN Question 3:")
scores = cross_val_scores(x_train, y_train)
print("Cross-validation scores for each fold:", scores)
print("Mean cross-validation score:", np.mean(scores))
print("Standard deviation of cross-validation scores:", np.std(scores))

# KNNQ4
print("\nKNN Question 4:")
KNN_Q4(x_train, y_train)

# CMQ1
print("\nConfusion Matrix Question 1:")
confussion_matrix(y_test, y_pred)

# DTCQ1
print("\nDecision Tree Classifier Question 1:")
accuracy_tree, report_tree = decision_tree_classifier(x_train, x_test, y_train, y_test)
print("Decision Tree Classifier Accuracy:", accuracy_tree)
print("Decision Tree Classification Report:\n", report_tree)

# LRQ1
print("\nLogistic Regression Question 1:")
logistic_regression(x_train_scaled, y_train)

# PCAQ1
print("\nPCA Question 1:")
pca_Q1()
print(f"Shape of X_digits: {X_digits.shape}")
print(f"Shape of images: {images.shape}")

# PCAQ2
print("\nPCA Question 2:")
scores, pca = pca_Q2()

# PCAQ3
print("\nPCA Question 3:")
pca_Q3(pca)

# PCAQ4
print("\nPCA Question 4:")
pca_Q4(scores, pca)