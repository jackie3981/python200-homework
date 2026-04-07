import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import flow, task, get_run_logger

# Paths and Constants
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "happiness_project")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
YEARS      = list(range(2015, 2025))


# Task 1: Load Multiple Years of Data 
@task(retries=3, retry_delay_seconds=2)
def load_data() -> pd.DataFrame:
    logger = get_run_logger()
    frames = []

    for year in YEARS:
        path = os.path.join(DATA_DIR, f"world_happiness_{year}.csv")
        df = pd.read_csv(path, sep=";", decimal=",")
        df["year"] = year
        df.rename(columns={"Happiness score": "happiness_score",
                            "Ladder score": "happiness_score"}, inplace=True)
        frames.append(df)
        logger.info(f"Loaded {year}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, "merged_happiness.csv")
    merged.to_csv(output_path, index=False)
    logger.info(f"Merged dataset saved → {output_path} ({len(merged)} total rows)")

    return merged

# Task 2: Descriptive Statistics 
@task
def descriptive_stats(df: pd.DataFrame) -> None:
    logger = get_run_logger()

    mean   = df["happiness_score"].mean()
    median = df["happiness_score"].median()
    std    = df["happiness_score"].std()

    logger.info(f"Overall Happiness Score — Mean: {mean:.4f} | Median: {median:.4f} | Std: {std:.4f}")

    logger.info(" -- Mean happiness score by year --")
    by_year = df.groupby("year")["happiness_score"].mean().sort_index()
    for year, score in by_year.items():
        logger.info(f"  {year}: {score:.4f}")

    logger.info(" -- Mean happiness score by region --")
    by_region = (
        df.groupby("Regional indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )
    for region, score in by_region.items():
        logger.info(f"  {region}: {score:.4f}")


# Task 3: Visual Exploration 
@task
def visual_exploration(df: pd.DataFrame) -> None:
    logger = get_run_logger()

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(df["happiness_score"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of Happiness Scores (2015–2024)")
    ax.set_xlabel("Happiness Score")
    ax.set_ylabel("Frequency")
    path = os.path.join(OUTPUT_DIR, "happiness_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")

    # Boxplot by year
    years_sorted = sorted(df["year"].unique())
    data_by_year = [df[df["year"] == y]["happiness_score"].dropna().values for y in years_sorted]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(data_by_year, tick_labels=years_sorted)
    ax.set_title("Happiness Score Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    path = os.path.join(OUTPUT_DIR, "happiness_by_year.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")

    # Scatter GDP vs Happiness
    fig, ax = plt.subplots()
    ax.scatter(df["GDP per capita"], df["happiness_score"], alpha=0.3, s=10, color="coral")
    ax.set_title("GDP per Capita vs Happiness Score")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    path = os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved → {path}")

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include="number").drop(columns=["Ranking", "year"])
    corr_matrix  = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# Task 4: Hypothesis Testing 
@task
def hypothesis_testing(df: pd.DataFrame) -> None:
    logger = get_run_logger()

    # Test 1: 2019 vs 2020 (pre/post pandemic)
    scores_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    scores_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020)

    logger.info(" -- T-Test: 2019 vs 2020 (Pre/Post Pandemic) --")
    logger.info(f"  Mean 2019: {scores_2019.mean():.4f}")
    logger.info(f"  Mean 2020: {scores_2020.mean():.4f}")
    logger.info(f"  T-statistic: {t_stat:.4f} | P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info(
            "  Result: The difference in happiness scores between 2019 and 2020 is "
            "statistically significant (p < 0.05). This suggests the pandemic onset "
            "was associated with a measurable change in global happiness."
        )
    else:
        logger.info(
            "  Result: No statistically significant difference was found between 2019 "
            "and 2020 happiness scores (p >= 0.05). The pandemic onset does not appear "
            "to have produced a detectable shift in global happiness at this threshold."
        )

    # Test 2: Western Europe vs Sub-Saharan Africa
    western_europe = df[df["Regional indicator"] == "Western Europe"]["happiness_score"].dropna()
    sub_saharan    = df[df["Regional indicator"] == "Sub-Saharan Africa"]["happiness_score"].dropna()

    t_stat2, p_value2 = stats.ttest_ind(western_europe, sub_saharan)

    logger.info(" -- T-Test: Western Europe vs Sub-Saharan Africa --")
    logger.info(f"  Mean Western Europe: {western_europe.mean():.4f}")
    logger.info(f"  Mean Sub-Saharan Africa: {sub_saharan.mean():.4f}")
    logger.info(f"  T-statistic: {t_stat2:.4f} | P-value: {p_value2:.4f}")

    if p_value2 < 0.05:
        logger.info(
            "  Result: The difference in happiness scores between Western Europe and "
            "Sub-Saharan Africa is statistically significant (p < 0.05), confirming "
            "a substantial regional gap in well-being."
        )
    else:
        logger.info(
            "  Result: No statistically significant difference found between Western Europe "
            "and Sub-Saharan Africa (p >= 0.05)."
        )


# Task 5: Correlation and Multiple Comparisons 
@task
def correlation_analysis(df: pd.DataFrame) -> dict:
    logger = get_run_logger()

    explanatory_vars = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
    ]

    results = {}
    logger.info(" -- Pearson Correlations with Happiness Score --")

    for var in explanatory_vars:
        clean = df[["happiness_score", var]].dropna()
        r, p = stats.pearsonr(clean["happiness_score"], clean[var])
        results[var] = {"r": r, "p": p}
        logger.info(f"  {var}: r={r:.4f} | p={p:.4f}")

    n_tests        = len(results)
    adjusted_alpha = 0.05 / n_tests

    logger.info(f" -- Multiple Comparisons: {n_tests} tests | Adjusted alpha: {adjusted_alpha:.4f} --")

    for var, vals in results.items():
        sig_original  = vals["p"] < 0.05
        sig_corrected = vals["p"] < adjusted_alpha
        status = (
            "significant (original + corrected)" if sig_corrected else
            "significant (original only)"         if sig_original  else
            "not significant"
        )
        logger.info(f"  {var}: {status}")

    return results


# Task 6: Summary Report 
@task
def summary_report(df: pd.DataFrame, correlation_results: dict) -> None:
    logger = get_run_logger()

    n_countries = df["Country"].nunique()
    n_years     = df["year"].nunique()
    logger.info(f"Total countries in dataset: {n_countries} | Years covered: {n_years} (2015–2024)")

    by_region = (
        df.groupby("Regional indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )
    top3    = by_region.head(3)
    bottom3 = by_region.tail(3)

    logger.info("Top 3 regions by mean happiness score:")
    for region, score in top3.items():
        logger.info(f"  {region}: {score:.4f}")

    logger.info("Bottom 3 regions by mean happiness score:")
    for region, score in bottom3.items():
        logger.info(f"  {region}: {score:.4f}")

    scores_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    scores_2020 = df[df["year"] == 2020]["happiness_score"].dropna()
    _, p_value  = stats.ttest_ind(scores_2019, scores_2020)

    if p_value < 0.05:
        logger.info(
            f"Pre/Post Pandemic (2019 vs 2020): Statistically significant difference found "
            f"(p={p_value:.4f}). The pandemic onset was associated with a measurable change "
            f"in global happiness scores."
        )
    else:
        logger.info(
            f"Pre/Post Pandemic (2019 vs 2020): No statistically significant difference found "
            f"(p={p_value:.4f}). Global happiness scores did not shift detectably at the pandemic onset."
        )

    adjusted_alpha = 0.05 / len(correlation_results)
    significant    = {var: vals for var, vals in correlation_results.items() if vals["p"] < adjusted_alpha}

    if significant:
        strongest = max(significant, key=lambda v: abs(significant[v]["r"]))
        logger.info(
            f"Strongest predictor of happiness (after Bonferroni correction): "
            f"'{strongest}' (r={significant[strongest]['r']:.4f})"
        )
    else:
        logger.info("No variables remained significant after Bonferroni correction.")


# Flow 
@flow(name="happiness_pipeline")
def happiness_pipeline():
    df                  = load_data()
    descriptive_stats(df)
    visual_exploration(df)
    hypothesis_testing(df)
    corr_results        = correlation_analysis(df)
    summary_report(df, corr_results)


if __name__ == "__main__":
    happiness_pipeline()