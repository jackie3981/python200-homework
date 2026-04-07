import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

# Data for the descriptive stats review
# DSQ1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

#DSQ3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

#DSQ4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

def numpy_stats():
    # DSQ1: Use NumPy to compute and print the mean, median, variance, and standard deviation.

    # Calculate mean
    mean = np.mean(data)
    
    # Calculate median
    median = np.median(data)
    
    # Calculate variance
    # Note: I used np.round() to round the random values to 4 decimal places, without the np.round() the decimal part is long.
    variance = np.round(np.var(data), 4)
    
    # Calculate standard deviation
    # Note: I used np.round() to round the random values to 4 decimal places, without the np.round() the decimal part is long.
    std_dev = np.round(np.std(data), 4)
    
    return mean, median, variance, std_dev

def distribution_mean_std():
    # DSQ2: Generate a normal distribution with a mean of 65 and a standard deviation of 10, and print the first 10 values.
    # Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.

    # Normal distribution with mean 65 and std deviation 10
    normal_dist = np.random.normal(loc=65, scale=10, size=500)

    plt.hist(normal_dist, bins=20, edgecolor='black')
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

    return normal_dist

def box_plot():
    # DSQ3: Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
    labels = ["Group A", "Group B"]

    plt.boxplot([group_a, group_b], labels=labels)
    plt.title("Score Comparison")
    plt.ylabel("Scores")
    plt.show()

def side_by_side_box_plot():
    # DSQ4: Create a side-by-side boxplot comparing the two distributions. Label each boxplot appropriately 
    # ("Normal" and "Exponential") and add a title "Distribution Comparison".
    # Add a comment in your code briefly noting which distribution is more skewed, and which descriptive 
    # statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

    # The exponential distribution is more skewed, as its values are concentrated near zero with a long tail to the right.
    # For the normal distribution, the mean is appropriate since it is symmetric. For the exponential distribution, 
    # the median is more appropriate since it is not affected by the long tail, unlike the mean.

    labels = ["Normal", "Exponential"]

    plt.boxplot([normal_data, skewed_data], labels=labels)
    plt.title("Distribution Comparison")
    plt.ylabel("Values")
    plt.show()

def mean_median_mode():
    # DSQ5: Print the mean, median, and mode. Why are the median and mean so different for data2? Add your answer as a comment in the code.

    # The mean and median are very different in data2 because of the outlier (150). The mean is sensitive to extreme values,
    # so the outlier pulls it up significantly. The median, however, only depends on the middle value, so it is not affected
    # by the outlier and remains a better measure of central tendency in this case.

    data1 = [10, 12, 12, 16, 18]
    data2 = [10, 12, 12, 16, 150]

    mean_data1 = np.mean(data1)
    mean_data2 = np.mean(data2)

    median1 = np.median(data1)
    median2 = np.median(data2)

    mode1 = stats.mode(data1)
    mode2 = stats.mode(data2)

    return (mean_data1, median1, mode1), (mean_data2, median2, mode2)

def descriptive_stats_review():
    print("DSQ1: Descriptive Statistics")
    mean, median, variance, std_dev = numpy_stats()
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")

    print("\nDSQ2: Distribution, Mean, and Standard Deviation")
    normal_dist = distribution_mean_std()
    print(f"First 10 values: {normal_dist[:10]}")

    print("\nDSQ3: Box Plot")
    box_plot()

    print("\nDSQ4: Side-by-Side Box Plot")
    side_by_side_box_plot()

    print("\nDSQ5: Mean, Median, and Mode")
    (mean1, med1, mod1), (mean2, med2, mod2) = mean_median_mode()
    print("Data 1:")
    print(f"  Mean: {mean1}, Median: {med1}, Mode: {mod1}")
    print("Data 2:")
    print(f"  Mean: {mean2}, Median: {med2}, Mode: {mod2}")

