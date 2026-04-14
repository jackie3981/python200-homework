import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#Path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

## The scikit-learn API Data
#SLQ1 Data
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

#SLQ2 Data
x = np.array([10, 20, 30, 40, 50])

#SLQ3 Data
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

## Linear Regression Data
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


## ===============================================
## The scikit-learn API
## ===============================================
def predict_salary(years_list):
    model = LinearRegression()
    model.fit(years, salary)
    print(f"Slope: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    for yr in years_list:
        pred = model.predict(np.array([[yr]]))[0]
        print(f"Predicted salary for {yr} years of experience: {pred}")

def array_2d():
    array_2d = x.reshape(-1, 1)
    print(f"Original shape: {x.shape}")
    print(f"New shape: {array_2d.shape}")

    # scikit-learn needs X to be 2D because it expects the input to be in the form of a matrix where rows represent samples 
    # and columns represent features. Even if there is only one feature, it still needs to be in a 2D format to maintain 
    # consistency and allow for multiple features in the future.   

def synthetic_dataset():
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_clusters)
    labels = kmeans.predict(X_clusters)
    print(f"Cluster centers:\n{kmeans.cluster_centers_}")
    print(f"Points in each cluster: {np.bincount(labels)}")
    plt.figure()
    plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=200, label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters.png'))
    plt.show()
        
## ===============================================
## Linear Regression
## ===============================================
def scatter_plot():
    plt.figure()
    plt.scatter(age, cost, c=smoker, cmap='coolwarm', edgecolor='k')
    plt.title('Medical Cost vs Age')
    plt.xlabel('Age')
    plt.ylabel('Cost')
    plt.legend(['Smoker','Non-smoker'])
    plt.savefig(os.path.join(OUTPUT_DIR, 'cost_vs_age.png'))
    plt.show()
    # the graphic shows a clear positive correlation between age and medical costs, as the points tend to rise as age increases.
    # additionally, the color coding indicates that smokers generally have higher medical costs compared to non-smokers at any given age.

def linear_regression_Q2():
    x_train, x_test, y_train, y_test = train_test_split(age.reshape(-1, 1), cost, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
     

def linear_regression_Q3(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f"Slope: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    y_pred = model.predict(x_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    r2 = model.score(x_test, y_test)
    print(f"RMSE: {rmse}")
    print(f"R² on the test set: {r2}")
    # Interpretation of the slope: The slope represents the average increase in medical costs for each 
    # additional year of age. For example, if the slope is 200, it means that on average, medical costs 
    # increase by $200 for every additional year of age, holding all other factors constant. 
    # This indicates that age is a significant factor in determining medical costs, and as people get older, 
    # their medical expenses tend to rise.

def linear_regression_smoker():
    x_full = np.column_stack([age, smoker])
    x_train, x_test, y_train, y_test = train_test_split(x_full, cost, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("age coefficient:    ", model.coef_[0])
    print("smoker coefficient: ", model.coef_[1])
    print(f"Intercept: {model.intercept_}")
    r2 = model.score(x_test, y_test)
    print(f"R² on the test set with smoker feature: {r2}")
    # the smoker coeficient represents the average increase in medical costs associated with being a smoker compared to a non-smoker,

def predicted_vs_actual(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.figure()
    plt.scatter(y_pred, y_test, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Predicted Cost')
    plt.ylabel('Actual Cost')
    plt.legend(['Data Points', 'Perfect Prediction'])
    plt.savefig(os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png'))
    plt.show()
    # a dot over the diagonal line indicates that the model underestimated the actual cost, 
    # while a dot under the diagonal means that the model overestimated the actual cost.
    
    
## Main execution block
# SLQ1
predict_salary([4, 8])

# SLQ2
array_2d()

# SLQ3
synthetic_dataset()

# LRQ1
scatter_plot()

# LRQ2
x_train, x_test, y_train, y_test = linear_regression_Q2()
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}") 

# LRQ3
linear_regression_Q3(x_train, x_test, y_train, y_test)

# LRQ4
linear_regression_smoker()

# LRQ5
x_full = np.column_stack([age, smoker])
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x_full, cost, test_size=0.2, random_state=42)
predicted_vs_actual(x_train_full, x_test_full, y_train_full, y_test_full)

