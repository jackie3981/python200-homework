import numpy as np
from scipy import stats

# Data for the hypothesis review
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

scores = [72, 68, 75, 70, 69, 74, 71, 73]


def independent_ttest():
    # HQ1: Run an independent samples t-test on group_a and group_b. Return t-statistic and p-value.
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    return t_stat, p_value


def significance_check(p_value, alpha=0.05):
    # HQ2: Check if the result is statistically significant at alpha=0.05. Return True if significant.
    return p_value < alpha


def paired_ttest():
    # HQ3: Run a paired t-test on before/after scores. Return t-statistic and p-value.
    t_stat, p_value = stats.ttest_rel(before, after)
    return t_stat, p_value


def onesample_ttest(benchmark=70):
    # HQ4: Run a one-sample t-test against a national benchmark of 70. Return t-statistic and p-value.
    t_stat, p_value = stats.ttest_1samp(scores, benchmark)
    return t_stat, p_value


def onetailed_ttest():
    # HQ5: Re-run HQ1 as a one-tailed test to check whether group_a scores are less than group_b.
    # Using the 'alternative' parameter for the one-tailed test.
    t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative='less')
    return t_stat, p_value


def conclusion(p_value):
    # HQ6: Return a plain-language conclusion for the result of HQ1.
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    diff = mean_b - mean_a
    if p_value < 0.05:
        return (f"The average score of Group B ({mean_b:.1f}) is significantly higher than "
                f"Group A ({mean_a:.1f}), with a difference of {diff:.1f} points. "
                f"The p-value ({p_value:.6f}) is well below 0.05, indicating that this difference "
                f"is extremely unlikely to be due to random chance alone.")
    else:
        return (f"The average score of Group B ({mean_b:.1f}) is higher than Group A ({mean_a:.1f}), "
                f"but the difference is not statistically significant (p = {p_value:.4f} > 0.05), "
                f"meaning it could reasonably be due to chance.")


def hypothesis_review():
    print("HQ1: Independent Samples T-Test")
    t_stat, p_value = independent_ttest()
    print(f"Group A: {group_a}")
    print(f"Group B: {group_b}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")

    print("\nHQ2: Statistical Significance Check")
    is_significant = significance_check(p_value)
    alpha = 0.05
    if is_significant:
        print(f"p-value ({p_value:.6f}) < alpha ({alpha}) -> Result is statistically significant")
    else:
        print(f"p-value ({p_value:.6f}) >= alpha ({alpha}) -> Result is NOT statistically significant")

    print("\nHQ3: Paired T-Test")
    t_stat3, p_value3 = paired_ttest()
    print(f"Before: {before}")
    print(f"After:  {after}")
    print(f"t-statistic: {t_stat3:.4f}")
    print(f"p-value: {p_value3:.6f}")

    print("\nHQ4: One-Sample T-Test")
    t_stat4, p_value4 = onesample_ttest()
    print(f"Scores: {scores}")
    print(f"Benchmark: 70")
    print(f"Sample mean: {np.mean(scores):.2f}")
    print(f"t-statistic: {t_stat4:.4f}")
    print(f"p-value: {p_value4:.6f}")

    print("\nHQ5: One-Tailed T-Test")
    t_stat5, p_value5 = onetailed_ttest()
    print(f"Group A mean: {np.mean(group_a):.2f}")
    print(f"Group B mean: {np.mean(group_b):.2f}")
    print(f"t-statistic: {t_stat5:.4f}")
    print(f"One-tailed p-value: {p_value5:.6f}")

    print("\nHQ6: Conclusion")
    print(conclusion(p_value))