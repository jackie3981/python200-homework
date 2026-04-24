import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  

# Paths and Constants
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Load the dataset
# The CSV uses semicolons (;) as field separators.
data_path = os.path.join(DATA_DIR, "student_performance_math.csv")
df = pd.read_csv(data_path, sep=";")



def load_explore():
    print("Shape of the dataset:", df.shape)
    print("\nFirst five rows of the dataset:")  
    print(df.head())
    print("\nData types of all columns:")
    print(df.dtypes)   

    fig,ax = plt.subplots(figsize=(10, 6)) 
    ax.hist(df['G3'], bins=21, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Final Math Grades")
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Frequency")
    fig.savefig(os.path.join(OUTPUT_DIR, "g3_distribution.png"))
    plt.close(fig)

def preprocess_data():
    # convert columns to numeric values
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
    df['internet'] = df['internet'].map({'no': 0, 'yes': 1})
    df['higher'] = df['higher'].map({'no': 0, 'yes': 1})
    df['activities'] = df['activities'].map({'no': 0, 'yes': 1})

    # Filter out rows where G3=0
    print("Shape of the dataset before filtering G3=0 rows:", df.shape)
    df_filtered = df[df['G3'] != 0]
    print("Shape of the dataset after filtering G3=0 rows:", df_filtered.shape)

    # G3=0 represent students who did not take the final exam. These students may have had various reasons for not taking the exam.
    # Including these rows could skew the model's predictions and lead to inaccurate conclusions about the factors influencing student performance,
    # as it may not accurately reflect the underlying relationships between the features and the final grade.

    corr_original = df['absences'].corr(df['G3'])
    corr_filtered = df_filtered['absences'].corr(df_filtered['G3'])
    print("Pearson correlation between absences and G3 in the original dataset:", corr_original)
    print("Pearson correlation between absences and G3 in the filtered dataset:", corr_filtered)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_filtered['absences'], df_filtered['G3'], alpha=0.6, color='red', label='Filtered Data')
    ax.scatter(df['absences'], df['G3'], alpha=0.3, color='blue', label='Original Data (includes G3=0)')
    ax.set_xlabel("Absences")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("Relationship between Absences and Final Grades")
    ax.legend()

    # adding annotation to show the correlation values for both the original and filtered datasets
    ax.annotate(
        f"r (original) = {corr_original:.3f}\nr (filtered) = {corr_filtered:.3f}",
        xy=(0.97, 0.97),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray')
    )

    fig.savefig(os.path.join(OUTPUT_DIR, "absences_vs_g3.png"))
    plt.close(fig)

    # The correlation between absences and G3 is stronger in the filtered dataset because students with G3=0 
    # (who did not take the final exam) tend to have high absences. Including them dilutes the negative 
    # relationship: they add high-absence / zero-grade points that are not real grades but exam absences, pulling the correlation toward zero.
    # After filtering, only students who actually took the exam remain, revealing a clearer negative relationship between absences and final grade.

    return df_filtered

    

def exploratory_data_analysis(df_filtered):
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        if col != 'G3':
            corr = df_filtered[col].corr(df_filtered['G3'])
            correlations[col] = corr
    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: item[1]))
    print("Pearson correlations with G3 (sorted):")
    for feature, corr in sorted_correlations.items():
        print(f"{feature}: {corr}")     
    # Among the model features (excluding G1 and G2), 'failures' has the strongest 
    # relationship with G3 (r = -0.294). On the positive side, 'Medu' shows the strongest positive correlation (r = 0.190).
    # Surprisingly, 'freetime' and 'activities' are nearly uncorrelated with G3.
    # G1 and G2 are included in the output but are not used as model features.


    # Figure 1: failures vs G3
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_filtered['failures'], df_filtered['G3'], alpha=0.5, color='green')
    ax.set_xlabel("Past Failures")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("Relationship between Past Failures and Final Grade")
    fig.savefig(os.path.join(OUTPUT_DIR, "failures_vs_g3.png"))
    plt.close(fig)
    # Students with more past failures tend to have lower final grades (r = -0.294).
    # This is the strongest predictor among the model features.

    # Figure 2: schoolsup vs G3 
    fig, ax = plt.subplots(figsize=(10, 6))
    df_filtered.boxplot(column='G3', by='schoolsup', ax=ax)
    ax.set_xlabel("School Support (0=No, 1=Yes)")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("G3 by School Support")
    plt.suptitle("")
    fig.savefig(os.path.join(OUTPUT_DIR, "schoolsup_vs_g3.png"))
    plt.close(fig)
    # students with school support (1) have lower median grades than those without (0). 
    # This reflects selection bias: support is assigned to students who are already struggling, not randomly distributed.


def baseline_model(df_filtered):
     
    X = df_filtered[['failures']]
    y = df_filtered['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    slope = model.coef_[0]  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Slope: {slope}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")  

    # On a 0-20 scale, the slope of -1.43 means each additional past failure is associated with a drop of about 1.4 grade points in the final grade.
    # The RMSE of 2.96 means the model's typical prediction error is roughly 3 points, which is significant on a 0-20 scale.
    # R² of 0.089 is low, meaning failures alone explains only 9% of the variance in G3.



def full_model(df_filtered, include_g1=False):
    feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]
    if include_g1:
        feature_cols.append("G1")

    X = df_filtered[feature_cols].values
    y = df_filtered["G3"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred) 
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Train R²: {train_r2}")
    print(f"Test R²: {test_r2}")
    print(f"Test RMSE: {rmse}")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"{name:12s}: {coef:+.3f}")

    # ****** Task 5 COMMENTS:  *********
    # Test R² improved from 0.089 (baseline) to 0.154 -- adding more features helps.
    # The model still explains only 15% of variance in G3.
    # Train R² (0.175) and Test R² (0.154) are close, indicating no significant overfitting.
    # Coefficient analysis:
    # - schoolsup (-2.062) is the largest negative coefficient -- students with school support score ~2 points lower. 
    # This reflects selection bias: support is given to students already struggling.
    # - internet (+0.834) is the largest positive coefficient 
    # Having internet access may reflect socioeconomic advantages that correlate with better performance.
    # - freetime (-0.042) and activities (-0.009) are nearly zero -- these features add almost no predictive value.
    # For production, I would keep: failures, schoolsup, internet, higher, studytime, sex.
    # I would drop: freetime, activities (near-zero coefficients) and traveltime (-0.112), they contribute very little to the model's predictions.

    # ****** Bonus with G1:  *********
    # Adding G1 jumps Test R² from 0.154 to 0.749 -- a massive improvement.
    # G1 coefficient (+0.885) is by far the strongest predictor in the model, meaning each additional point in G1 adds 0.9 points to the predicted G3.

    # Does a high R² mean G1 causes G3? No -- correlation is not causation.
    # G1 and G3 are both reflections of the same underlying student -- their ability, motivation, and circumstances. G1 is a proxy for everything we couldn't measure.

    # Is this a useful model for identifying students who might struggle?
    # Only partially -- it can flag students after G1 is available (mid-year), but by then it may be too late for early intervention.

    # What would educators need to do to intervene BEFORE G1 is available?
    # They would need to rely on the behavioral and demographic features from Task 5, accepting a weaker model (R²=0.154) in exchange for earlier intervention.
    # The tradeoff is: more time to help vs less accurate predictions.

    # adding G1 also changes other coefficients significantly -- some flip signs (e.g. higher went from +0.610 to -0.207, studytime from +0.448 to -0.113).
    # This is because G1 absorbs most of the variance those features were explaining before.

    return y_test, y_test_pred 

def evaluate_model(y_test, y_test_pred, df_filtered):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_pred, y_test, alpha=0.5, color='steelblue')
    ax.plot([0, 20], [0, 20], color='red', linestyle='--', label='Perfect Prediction')
    ax.set_xlabel("Predicted Grade")
    ax.set_ylabel("Actual Grade")
    ax.set_title("Predicted vs Actual (Full Model)")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"))
    plt.close(fig)

    # The model predicts almost everything between 9-13, regardless of actual grade.
    # Points above the diagonal: model underpredicted (actual > predicted) -- high-achieving students (15-19) are consistently underpredicted.
    # Points below the diagonal: model overpredicted (actual < predicted) -- low-achieving students (5-7) are consistently overpredicted.
    # Error is NOT uniform -- the model struggles most at both extremes, and is most reliable for average students (10-12).

    print(f"Filtered dataset: {len(df_filtered)} students; test set: {len(y_test)} students")
    # RMSE of 2.86 means a typical prediction is off by roughly 3 points on a 0-20 scale,
    # which represents about 15% of the total grade range -- a significant margin of error.
    # R² of 0.154 means the model explains only 15% of the variance in final grades.
    # This is a weak model -- most of what drives a student's grade is not captured by the behavioral and demographic features used here.
    # Largest positive coefficient: internet (+0.834) -- students with internet access at home score about 0.8 points higher, likely reflecting broader socioeconomic advantage.
    # Largest negative coefficient: schoolsup (-2.062) -- students receiving school support score about 2 points lower, reflecting selection bias: support targets struggling students.
    # Surprising result: internet access is a stronger predictor than wanting to pursue higher education (higher: +0.610), suggesting that material resources matter
    # more than academic aspirations in this dataset.
  
# Task 1: Load and Explore the Dataset
print("Loading and exploring the dataset")
load_explore()

# Task 2: Preprocess the Data
print("\nPreprocessing the data")
df_filtered = preprocess_data()

# Task 3: Exploratory Data Analysis
print("\nPerforming exploratory data analysis")
exploratory_data_analysis(df_filtered)

# Task 4: Baseline Model
print("\nBuilding and evaluating the baseline model")
baseline_model(df_filtered)

# Task 5: Full Model
print("\nBuilding and evaluating the full model")
y_test, y_test_pred = full_model(df_filtered, include_g1=False)

# Task 6: Evaluate and Summarize
print("\nEvaluating the full model")
evaluate_model(y_test, y_test_pred, df_filtered) 

# Bonus: Add G1 to the full model
print("\nBuilding full model with G1")
full_model(df_filtered, include_g1=True)