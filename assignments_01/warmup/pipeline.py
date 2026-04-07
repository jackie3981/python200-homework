import numpy as np
import pandas as pd

arrQ1 = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


# ============================================================================
# PIPELINE QUESTION 1: Plain Python Pipeline
# ============================================================================

def create_series(arr):
    """
    Takes a NumPy array and returns a pandas Series with the name "values"
    """
    return pd.Series(arr, name="values")


def clean_data(series):
    """
    Takes the Series, removes any NaN values using .dropna(),
    and returns the cleaned Series
    """
    return series.dropna()


def summarize_data(series):
    """
    Takes the cleaned Series and returns a dictionary with four keys:
    "mean", "median", "std", and "mode"
    For mode, uses series.mode()[0] to get a single value
    """
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0] if len(series.mode()) > 0 else np.nan
    }


def data_pipeline(arr):
    """
    Calls the three functions above in sequence and returns the summary dictionary
    """
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


def pipeline_review():
   
    print("PQ1 - Plain Python Pipeline")
    result_pq1 = data_pipeline(arrQ1)
    print(f"Original array with NaNs: {arrQ1}")
    print(f"Number of elements: {len(arrQ1)}")
    print(f"Number of NaNs: {np.isnan(arrQ1).sum()}")
    print("\nSummary dictionary after pipeline:")
    for key, value in result_pq1.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print()