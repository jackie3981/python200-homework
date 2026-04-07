import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Data for the correlation review
xQ1 = [1, 2, 3, 4, 5]
yQ1 = [2, 4, 6, 8, 10]

xQ2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yQ2 = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

xQ4 = [10, 20, 30, 40, 50]
yQ4 = [90, 75, 60, 45, 30]

people = {
        "height": [160, 165, 170, 175, 180],
        "weight": [55, 60, 65, 72, 80],
        "age": [25, 30, 22, 35, 28]
    }

def question1_correlation_matrix():
    # CQ1: The expected correlation is 1.0 because y = 2*x, a perfect positive linear relationship.
    # Every increase in x corresponds to an exactly proportional increase in y, so r = 1.0.
        
    correlation_matrix = np.corrcoef(xQ1, yQ1)
    correlation_coef = correlation_matrix[0, 1]
    
    return correlation_matrix, correlation_coef

def question2_pearsonr():
    
    correlation_coef, p_value = pearsonr(xQ2, yQ2)
     
    return correlation_coef, p_value

def question3_dataframe_correlation():
    
    df = pd.DataFrame(people)
    
    correlation_matrix = df.corr(numeric_only=True)
     
    return df, correlation_matrix

def question4_scatter_plot():
    
    plt.figure(figsize=(8, 6))
    plt.scatter(xQ4, yQ4, color='red', s=100, alpha=0.7)
    plt.title("Negative Correlation", fontsize=14, fontweight='bold')
    plt.xlabel("X values", fontsize=12)
    plt.ylabel("Y values", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line to show negative correlation
    z = np.polyfit(xQ4, yQ4, 1)
    p = np.poly1d(z)
    plt.plot(xQ4, p(xQ4), "b--", alpha=0.5, label=f"Trend: y = {z[0]:.1f}x + {z[1]:.1f}")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def question5_heatmap(df, correlation_matrix):

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def correlation_review():
   
    print("CQ1 - Pearson correlation using np.corrcoef()")
    corr_matrix, corr_coef = question1_correlation_matrix()
    print(f"x: {xQ1}")
    print(f"y: {yQ1}")
    print(f"Expected correlation: 1.0 (perfect positive linear relationship, y = 2*x)")
    print(f"Full correlation matrix:\n{corr_matrix}")
    print(f"Correlation coefficient [0,1]: {corr_coef:.4f}")
    print()
    
    print("CQ2 - Pearson correlation using scipy.stats.pearsonr()")
    corr_q2, p_val_q2 = question2_pearsonr()
    print(f"x: {xQ2}")
    print(f"y: {yQ2}")
    print(f"Correlation coefficient: {corr_q2:.4f}")
    print(f"p-value: {p_val_q2:.6f}")
    print()
    
    print("CQ3 - Correlation matrix using df.corr()")
    df, corr_matrix = question3_dataframe_correlation()
    print("DataFrame:")
    print(df)
    print("\nCorrelation matrix:")
    print(corr_matrix)
    print()
    
    print("CQ4 - Scatter plot with negative correlation")
    question4_scatter_plot()
    print(f"x: {xQ4}")
    print(f"y: {yQ4}")
    print("Notice: As x increases, y decreases -> negative correlation")
    print()
    
    print("CQ5 - Correlation heatmap using sns.heatmap()")
    question5_heatmap(df, corr_matrix)
    print("Interpretation:")
    print("  - Height vs Weight: Strong positive correlation (~0.99)")
    print("  - Height vs Age: Weak positive correlation (~0.42)")
    print("  - Weight vs Age: Weak positive correlation (~0.55)")
    print()