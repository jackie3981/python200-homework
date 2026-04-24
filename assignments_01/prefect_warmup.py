import numpy as np
import pandas as pd
from prefect import flow, task

@task
def create_series_task(arr):
    return pd.Series(arr, name="values")


@task
def clean_data_task(series):
    return series.dropna()


@task
def summarize_data_task(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0] if len(series.mode()) > 0 else np.nan
    }


@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series_task(arr)
    cleaned_series = clean_data_task(series)
    summary = summarize_data_task(cleaned_series)
    return summary

def prefect_pipeline_review():
    result = pipeline_flow()
    print(f"Original array with NaNs: [12.0, 15.0, nan, 14.0, 10.0, nan, 18.0, 14.0, 16.0, 22.0, nan, 13.0]")
    print(f"\nSummary dictionary after Prefect pipeline:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    pipeline_flow()


# ============================================================================
# Discussion Questions
# ============================================================================

# Q: This pipeline is simple -- just three small functions on a handful of numbers.
#    Why might Prefect be more overhead than it is worth here?
#
# A: Prefect adds significant overhead for such a simple pipeline because:
#    - The pipeline runs in milliseconds, but Prefect adds orchestration latency
#    - No scheduling, retries, or monitoring are needed for a one-off script
#    - Installing and configuring Prefect adds complexity without any real benefit


# Q: Describe some realistic scenarios where a framework like Prefect could still
#    be useful, even if the pipeline logic itself stays simple.
#
# A: Even with simple logic, Prefect is valuable when:
#    1. The pipeline runs on a schedule (daily/weekly reports)
#    2. Tasks may fail intermittently (API calls, network issues) and need retries
#    3. Parallel processing of many files or partitions is needed
#    4. The pipeline runs in production where visibility into failures is critical
#    5. Caching is required to avoid repeating expensive operations
#    6. The pipeline needs to be parameterized and reused across different datasets