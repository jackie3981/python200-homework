import pandas as pd

# Data for the pandas review
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

## Pandas Review

def first_three_rows():
    # PQ1: print the first 3 rows of the dataframe
    return df.head(3)
    #print(dfQ1)

def passed_students_above_80(dfQ1):
    # PQ2: print the names of all students who passed and have a grade above 80, using dataframe from Q1
    passed_students = dfQ1[(dfQ1["passed"] == True) & (dfQ1["grade"] > 80)]["name"]
    return passed_students

def add_grade_curved():
    # PQ3: Add a new column called "grade_curved" that adds 5 points to each student's grade.
    df["grade_curved"] = df["grade"] + 5
    return df

def add_name_upper():
    # PQ4: Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor.
    df["name_upper"] = df["name"].str.upper()
    return df

def mean_grade_by_city():
    # PQ5: Group the DataFrame by "city" and compute the mean grade for each city
    grade_by_city_mean = df.groupby("city")["grade"].mean()
    return grade_by_city_mean

def replace_austin_with_houston():
    # PQ6: Replace the value "Austin" in the "city" column with "Houston".
    df["city"] = df["city"].replace("Austin", "Houston")
    return df

def sort_by_grade_descending():
    # PQ7: Sort the DataFrame by "grade" in descending order and print the top 3 rows.
    df_sorted = df.sort_values("grade", ascending=False)
    return df_sorted.head(3)

def pandas_review():
    print("Q1:")
    print(first_three_rows())
    print("\nQ2:")
    print(passed_students_above_80(df))
    print("\nQ3:")
    print(add_grade_curved())
    print("\nQ4:")
    print(add_name_upper())
    print("\nQ5:")
    print(mean_grade_by_city())
    print("\nQ6:")
    print(replace_austin_with_houston())
    print("\nQ7:")
    print(sort_by_grade_descending())
