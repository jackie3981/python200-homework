import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

# Initial Setup
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = Path(BASE_DIR).parent.joinpath("assignments_01", "outputs", "merged_happiness.csv")

api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
When you save a plot successfully (no error), immediately call final_answer() 
with a short confirmation message. Do not attempt to run any more code.
Be concise and student-friendly in your responses.

Important: The column names in the dataset are:
- happiness_score (not 'Happiness Score')
- GDP per capita (not 'gdp_per_capita')
- Regional indicator (not 'region')
"""

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between GDP per capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    f"Plot happiness_score over the years as a line chart, with one line per region. Save the plot to {os.path.join(BASE_DIR, 'outputs', 'happiness_by_region.png')}.",
]

responses = []

# Data manager — holds the DataFrame so the agent sandbox can access it
class HappinessDataManager:
    def __init__(self):
        self.df = None

data_manager = HappinessDataManager()

# *** Task 1: Define Your Tools ***
@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.

    The dataset is loaded from DATA_PATH (merged_happiness.csv).
    After loading, the DataFrame is stored in data_manager.df.

    Returns:
        dict: A dictionary with keys "shape" (tuple) and "columns" (list),
              or an error dict if the file is not found.
    """
    if not DATA_PATH.exists():
        return {"error": f"Data file not found at {DATA_PATH}. Please run the merge script first."}

    data_manager.df = pd.read_csv(DATA_PATH)

    return {
        "shape": data_manager.df.shape,
        "columns": data_manager.df.columns.tolist(),
    }

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.

    Uses pandas.describe() to generate count, mean, std, min, quartiles, and max.

    Args:
        column: Name of the column to summarize (must be numeric).

    Returns:
        dict: Descriptive statistics, or an error dict if no data is loaded or column is not found.
    """
    if data_manager.df is None:
        return {"error": "No data loaded. Call load_happiness_data() first."}

    if column not in data_manager.df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {data_manager.df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(data_manager.df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    summary = data_manager.df[column].describe().to_dict()
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()}

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: First column name for correlation analysis.
        col2: Second column name for correlation analysis.

    Returns:
        dict: Dictionary with keys "col1", "col2", "pearson_r", and "p_value"
              (rounded to 4 decimal places), or an error dict on bad input.
    """
    if data_manager.df is None:
        return {"error": "No data loaded. Call load_happiness_data() first."}

    for col in [col1, col2]:
        if col not in data_manager.df.columns:
            return {"error": f"Column '{col}' not found. Available columns: {data_manager.df.columns.tolist()}"}
        if not pd.api.types.is_numeric_dtype(data_manager.df[col]):
            return {"error": f"Column '{col}' is not numeric."}

    valid_data = data_manager.df[[col1, col2]].dropna()

    if len(valid_data) < 3:
        return {"error": f"Insufficient data points: {len(valid_data)} rows after dropping NaN."}

    r, p_value = pearsonr(valid_data[col1], valid_data[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p_value), 4),
    }

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.

    Args:
        column: Column name to rank by (must be numeric).
        year: Year to filter the data.
        n: Number of top countries to return (default: 5).

    Returns:
        dict: Dictionary with "year", "column", and "top_countries" (list of dicts),
              or an error dict on bad input.
    """
    if data_manager.df is None:
        return {"error": "No data loaded. Call load_happiness_data() first."}

    if column not in data_manager.df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {data_manager.df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(data_manager.df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    if 'year' not in data_manager.df.columns:
        return {"error": "No 'year' column found in the dataset."}

    if 'Country' not in data_manager.df.columns:
        return {"error": "No 'Country' column found in the dataset."}

    df_year = data_manager.df[data_manager.df['year'] == year]

    if df_year.empty:
        return {"error": f"No data found for year {year}. Available years: {sorted(data_manager.df['year'].unique())}"}

    top_n = df_year.nlargest(n, column)[['Country', column]]

    return {
        "year": year,
        "column": column,
        "top_countries": [
            {"Country": row['Country'], column: row[column]}
            for _, row in top_n.iterrows()
        ],
    }

# Tools list
TOOLS = [
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries,
]

# Model and Agent setup
model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id="gpt-4o-mini",
)

# *** Task 2 / Task 3 / Task 4: Build the Agent ***
agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats","os"],
    max_steps=8,
)

def task_1():
    result = load_happiness_data()
    print("load_happiness_data:", result)

    if "error" not in result:
        print("\nsummarize_column('happiness_score'):", summarize_column("happiness_score"))
        print("\ncompute_correlation('happiness_score', 'GDP per capita'):",
              compute_correlation("happiness_score", "GDP per capita"))
        print("\nget_top_n_countries('happiness_score', 2019, n=3):",
              get_top_n_countries("happiness_score", 2019, n=3))

def task_2():
    print("Task 2 - Testing CodeAgent")
    response = agent.run(
        "Load the happiness data and show me the top 5 happiest countries in 2023",
        additional_args={"data_manager": data_manager}
    )
    print("\nAgent response:\n", response)

def task_3():
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)

        response = agent.run(
            query,
            reset=False,
            additional_args={"data_manager": data_manager}
        )
        responses.append(response)

        print(f"\nResponse {i}:")
        print(response)
        print("-"*60)

def task_4():
    query_1 = "What is the correlation between Social support and happiness_score? Is it statistically significant?"
    query_2 = "Which country has the highest average happiness_score across all years?"

    print(f"\nQuery 1: {query_1}")
    answer_1 = agent.run(query_1, reset=False)
    print(f"Answer query 1: {answer_1}")
    # Query 1: Social support has a strong positive correlation with happiness_score (pearson_r=0.7439), which is statistically significant (p=0.0).
    # This suggests that feeling socially connected is an important predictor of happiness.

    print(f"\nQuery 2: {query_2}")
    answer_2 = agent.run(query_2, reset=False)
    print(f"Answer query 2: {answer_2}")
    # Query 2: The agent used get_top_n_countries for year=2023 instead of computing the average across all years,
    # it got the right answer (Finland) by coincidence, but the approach was incorrect for the question asked.
    # A correct solution would require the agent to write code: df.groupby('Country')['happiness_score'].mean().idxmax()

def task_5():
    # --- Reflection ---
    #
    # 1. In Query 3, how did the agent communicate whether the correlation was statistically
    #    significant? Did it use the p-value correctly? What threshold did it apply?
    # 
    #    In Query 3, the agent returned both the Pearson correlation coefficient (0.6313) and the p-value (0.0), 
    #    explicitly stating the result was "statistically significant."
    #    It correctly applied the standard significance threshold of 0.05, a p-value of 0.0 (meaning p < 0.0001 in practice) 
    #    is well below that threshold.
    #    The agent interpreted the p-value correctly and communicated it clearly.

    # 2. Did any of the agent's responses surprise you — either by being more capable than
    #    you expected, or less? Describe one specific example.
    #
    #    The agent surprised me by showing it can recover from its own errors. In Task 3 Query 1, it tried several 
    #    approaches to access the DataFrame, including globals(), locals(), and try/except blocks -- before finding the
    #    correct solution. However, in Task 4 Query 2, it completely misread the instruction "across all years" and 
    #    focused only on 2023.
    #    This showed me that while the agent is quite good at following literal instructions, it can sometimes miss the 
    #    bigger picture or take shortcuts when an easier path exists using the available tools.

    # 3. What one additional tool would make this agent meaningfully more useful?
    #    Describe what it would do and what kind of question it would help the agent answer.
    #    (You do not need to implement it.)
    #
    #    Tool name: compare_countries_over_time
    #    This tool would take two country names and a column (e.g., happiness_score) and return a line chart or dictionary 
    #    showing how both countries evolved year by year. It could also compute the difference between them and indicate which
    #    country has consistently maintained higher values.
    #
    #    Example question it would help answer:
    #    "How has Finland's happiness_score compared to Denmark's over the years?
    #    Which country has been happier on average? Has the gap widened or narrowed?"
    #
    #    Currently this requires the agent to write custom pandas and matplotlib code to filter, group, merge, and plot. 
    #    A dedicated tool would make this much more efficient and less error-prone.
        
    pass

if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()