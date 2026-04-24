import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# fetch dataset
spambase = fetch_ucirepo(id=94)

X = spambase.data.features
y = spambase.data.targets

target_col = y.columns[0]

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y[target_col],
        test_size=0.2,
        random_state=42,
        stratify=y[target_col]
    )

def stats_1():
    df = pd.concat([X, y], axis=1)
    # target_col eliminado aqui -- ya esta definido a nivel de modulo

    features = [
        "word_freq_free",
        "char_freq_!",
        "capital_run_length_total"
    ]

    for feature in features:
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=feature, by=target_col, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("Class")
        plt.suptitle("")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{feature}.png"))
        plt.close(fig)

    # Class distribution: 2788 ham (60.6%) and 1813 spam (39.4%).
    # The dataset is moderately imbalanced -- a model that predicts "ham" for everything
    # would already achieve ~60% accuracy, so raw accuracy alone is not a reliable metric.

    # Boxplot observations:
    # - word_freq_free: spam emails show a higher median and wider spread than ham,
    #   but both distributions are heavily compressed near zero with many outliers.
    #   The difference is noticeable but not dramatic.
    # - char_freq_!: spam shows a slightly higher median than ham, but again both
    #   are near zero. The difference is subtle -- most emails use few exclamation marks.
    # - capital_run_length_total: spam has a visibly higher median and far more extreme
    #   outliers (up to ~15000) compared to ham. This is the most dramatic difference
    #   of the three features.

    # Why so many zeros? Most emails simply do not contain words like "free" at all.
    # These are frequency features -- zero means the word never appeared in that email.
    # The heavy skew toward zero is expected and normal for text-based features.

    # Why does the scale vary so much across features?
    # Word frequencies are tiny fractions (0-20%), while capital_run_length_total
    # can reach thousands. This matters for distance-based models like KNN, which
    # would be dominated by large-scale features. Tree-based models are not affected.

    return y, df

def task_02():
    # Scale first -- PCA finds directions of maximum variance, so features with
    # larger raw values would dominate without standardization.
    # Fit only on training data to prevent test-set information from leaking
    # into the components.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA()
    pca.fit(X_train_scaled)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(explained_variance)
    ax.axhline(y=0.90, linestyle="--")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Variance explained")
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_variance.png"))
    plt.close(fig)

    # Find n components for 90%
    n = np.argmax(explained_variance >= 0.90) + 1

    X_train_pca = pca.transform(X_train_scaled)[:, :n]
    X_test_pca = pca.transform(X_test_scaled)[:, :n]

    # Keep both full scaled arrays and PCA-reduced arrays:
    # - scaled arrays for models that don't benefit from PCA (logistic regression, KNN full)
    # - PCA arrays for models that benefit from reduced dimensionality
    return X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, n

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)

    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)

    print(f"\n{name}")
    print("Train accuracy:", accuracy_score(y_tr, train_pred))
    print("Test accuracy:", accuracy_score(y_te, test_pred))
    print(classification_report(y_te, test_pred))

def task_03(X_train_scaled, X_test_scaled, X_train_pca, X_test_pca):
    # -----------------
    # KNN
    # -----------------
    evaluate("KNN (unscaled)", KNeighborsClassifier(5),
             X_train, X_test, y_train, y_test)

    evaluate("KNN (scaled)", KNeighborsClassifier(5),
             X_train_scaled, X_test_scaled, y_train, y_test)

    evaluate("KNN (PCA)", KNeighborsClassifier(5),
             X_train_pca, X_test_pca, y_train, y_test)

    # -----------------
    # Decision Tree
    # -----------------
    depths = [3, 5, 10, None]

    best_depth = None
    best_score = 0

    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=42)
        tree.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, tree.predict(X_train))
        test_acc = accuracy_score(y_test, tree.predict(X_test))

        print(f"\nTree depth={d} | Train accuracy={train_acc:.3f} | Test accuracy={test_acc:.3f}")

        if test_acc > best_score:
            best_score = test_acc
            best_depth = d
    # *********************
    # As depth increases, train accuracy rises toward 1.0 while test accuracy plateaus:
    # depth=3: train=0.896, test=0.885 -- underfitting, too simple
    # depth=5: train=0.923, test=0.899 -- good balance
    # depth=10: train=0.967, test=0.909 -- slight overfitting starting
    # depth=None: train=1.000, test=0.911 -- clear overfitting, memorizes training data
    # The code selects depth=None as it has the highest test accuracy (0.911).
    # However, for production depth=5 would be preferable -- it achieves 0.899
    # with far less overfitting (train=0.923 vs train=1.000 for None).
    # The gain from None over depth=5 is only 0.002, not worth the fragility.

    tree_final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    evaluate("Decision Tree (final)", tree_final,
             X_train, X_test, y_train, y_test)

    # -----------------
    # Random Forest
    # -----------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    evaluate("Random Forest", rf,
             X_train, X_test, y_train, y_test)

    # -----------------
    # Logistic Regression
    # -----------------
    log_scaled = LogisticRegression(max_iter=1000, solver='liblinear')
    log_pca = LogisticRegression(max_iter=1000, solver='liblinear')

    evaluate("LogReg (scaled)", log_scaled,
             X_train_scaled, X_test_scaled, y_train, y_test)

    evaluate("LogReg (PCA)", log_pca,
             X_train_pca, X_test_pca, y_train, y_test)

    # -----------------
    # Feature Importances
    # -----------------
    tree_final2 = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    tree_final2.fit(X_train, y_train)

    rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf2.fit(X_train, y_train)

    feature_names = X.columns

    print("\nTop 10 features -- Decision Tree:")
    tree_importances = pd.Series(tree_final2.feature_importances_, index=feature_names)
    print(tree_importances.nlargest(10))

    print("\nTop 10 features -- Random Forest:")
    rf_importances = pd.Series(rf2.feature_importances_, index=feature_names)
    print(rf_importances.nlargest(10))

    fig, ax = plt.subplots(figsize=(15, 6))
    rf_importances.nlargest(10).sort_values().plot.barh(ax=ax)
    ax.set_title("Top 10 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"), bbox_inches= 'tight')
    plt.close(fig)

    # -----------------
    # Confusion Matrix (RF)
    # -----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(rf2, X_test, y_test, ax=ax)
    fig.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"))
    plt.close(fig)

    #*********************************
    # Summary:
    # Random Forest is the best model (test accuracy=0.944, precision spam=0.95).
    # KNN: scaling jumps accuracy from 0.799 to 0.908 -- confirms scale sensitivity.
    #   PCA (0.907) performs almost identically to scaled (0.908) -- no meaningful gain.
    # LogReg: scaled (0.929) outperforms PCA (0.919) -- dimensionality reduction
    #   slightly hurts logistic regression on this dataset.
    # Decision Tree and Random Forest agree on top features: char_freq_$, word_freq_remove,
    #   char_freq_! -- dollar signs and removal requests are the strongest spam signals.
    # For a spam filter, precision is the most important metric -- a false positive
    #   (legitimate email marked as spam) is more costly than a false negative.
    #   Random Forest achieves precision=0.95 for spam, the best of all models.

    return best_depth

def task_04(X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, best_depth):   

    models = [
        ("KNN (unscaled)", KNeighborsClassifier(5), X_train),
        ("KNN (scaled)", KNeighborsClassifier(5), X_train_scaled),
        ("KNN (PCA)", KNeighborsClassifier(5), X_train_pca),
        ("Decision Tree (final)", DecisionTreeClassifier(max_depth=best_depth, random_state=42), X_train),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train),
        ("LogReg (scaled)", LogisticRegression(max_iter=1000, solver='liblinear'), X_train_scaled),
        ("LogReg (PCA)", LogisticRegression(max_iter=1000, solver='liblinear'), X_train_pca),
    ]

    print("\nCross-Validation Results (cv=5):")
    for name, model, X_tr in models:
        scores = cross_val_score(model, X_tr, y_train, cv=5)
        print(f"{name:30s} | Mean: {scores.mean():.3f} | Std: {scores.std():.3f}")

    # Cross-validation confirms the Task 3 ranking -- no lucky draw from a single split.
    # Random Forest is the most accurate (mean=0.954) and reasonably stable (std=0.014).
    # Decision Tree has the highest variance (std=0.021) -- small changes in the data
    # splits produce noticeably different trees, which Random Forest averages away.
    # KNN unscaled is the weakest model (mean=0.794), confirming scale sensitivity.
    # KNN scaled and PCA produce nearly identical CV results (0.905 vs 0.908).
    # LogReg PCA (std=0.003) is the most stable model, but at the cost of lower accuracy.
    # The CV ranking matches the single train/test split from Task 3 -- results are reliable.

def task_05(n):
 
    # Best tree-based: Random Forest (no scaler needed -- trees are scale-invariant)
    rf_pipeline = Pipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Best non-tree: Logistic Regression scaled
    # Task 3 showed scaled (0.929) > PCA (0.919), so no PCA step needed
    logreg_pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    for name, pipeline in [("Random Forest Pipeline", rf_pipeline),
                            ("LogReg Pipeline", logreg_pipeline)]:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n{name}")
        print(classification_report(y_test, y_pred))

    # Pipeline structure comparison:
    # - RF pipeline has only one step (classifier) - no preprocessing needed. Trees split on thresholds, so feature scale does not affect the result.
    # - LogReg pipeline has two steps (scaler + classifier) - scaling is required because logistic regression is sensitive to feature magnitudes.

    # Practical value of pipelines:
    # A pipeline bundles preprocessing and model into a single object.
    # When handing off to another person or deploying to production, there is no risk of forgetting to scale new data or applying steps in the wrong order.
    # pipeline.predict(new_data) handles everything automatically and correctly.
    # Results should match Task 3 exactly - same logic, just packaged more cleanly.

    # Results confirmed - match Task 3 exactly:
    # Random Forest: accuracy=0.94, precision spam=0.95 
    # LogReg scaled: accuracy=0.93, precision spam=0.92 


# Task 1: Load and Explore
y, df = stats_1()
print("Shape:", df.shape)
print("\nClass distribution (count):")
print(y.value_counts())
print("\nClass distribution (proportion):")
print(y.value_counts(normalize=True))
print("\nStadistic resumen:")
print(df.describe())

# Task 2: Prepare Your Data
print("\nPCA preprocessing")
X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, n = task_02()
print("\nNumber of components for 90% variance:", n)

# Task 3: A Classifier Comparison
print("\nClassifier Comparison")
best_depth = task_03(X_train_scaled, X_test_scaled, X_train_pca, X_test_pca)

# Task 4: Cross-Validation
print("\nCross-Validation")
task_04(X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, best_depth)

# Task 5: Building a Prediction Pipeline
print("\nBuilding prediction pipelines")
task_05(n)