import matplotlib.pyplot as plt

# Data for the matplotlib review
# MQ1 - Line plot data
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# MQ2 - Bar plot data
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# MQ3 - Scatter plot data
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]



def line_plot():
    # MQ1: Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
    plt.plot(x, y)
    plt.title('Squares')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def bar_plot():
    # MQ2: Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
    plt.bar(subjects, scores)
    plt.title('Subject Scores')
    plt.xlabel('Subjects')
    plt.ylabel('Scores')
    plt.grid(axis='y')
    plt.show()

def scatter_plot():
    # MQ3: Plot the two datasets as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
    plt.scatter(x1, y1, c='blue', label='Dataset 1')
    plt.scatter(x2, y2, c='red', label='Dataset 2')
    plt.title('Scatter Plot Example')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def subplot_example():
    # MQ4: Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from MQ1 as a line. In the right subplot, plot the subjects and scores from MQ2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Line plot in the left subplot
    axs[0].plot(x, y)
    axs[0].set_title('Squares')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid()

    # Bar plot in the right subplot
    axs[1].bar(subjects, scores)
    axs[1].set_title('Subject Scores')
    axs[1].set_xlabel('Subjects')
    axs[1].set_ylabel('Scores')
    axs[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

def matplotlib_review():
    print("Line Plot:")
    line_plot()
    print("\nBar Plot:")
    bar_plot()
    print("\nScatter Plot:")
    scatter_plot()
    print("\nSubplot Example:")
    subplot_example()