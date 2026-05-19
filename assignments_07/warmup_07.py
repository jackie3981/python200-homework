from dotenv import load_dotenv
from openai import OpenAI
import os, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

# --- Setup ---

BASE_DIR = os.path.dirname(__file__)
RESOURCES_DIR = Path(BASE_DIR).joinpath("resources")

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

# =============================================================================
# Lesson 02: Tool Definitions and the ReAct Loop
# =============================================================================

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

def get_schema():
    """Return the tool schema list (get_current_time + celsius_to_fahrenheit)."""
    return [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_time',
                'description': 'Returns the current local time as a string.',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': [],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'celsius_to_fahrenheit',
                'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        "celsius": {
                            "type": 'number',
                            "description": 'Temperature in degrees Celsius'
                        }
                    },
                    'required': ['celsius'],
                },
            },
        }
    ]

# Q2 / Q3

def get_current_time():
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def run_agent(user_prompt: str):
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    AGENT_SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                celsius = arguments.get('celsius')
                tool_result = celsius_to_fahrenheit(celsius)
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    return first_message.content or ''

# =============================================================================
# Lesson 03: Multi-Tool Agent
# =============================================================================

# Q4 / Q5 / Q6

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Internal helpers ---

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools ---

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary stats for one or more columns."""
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot from the active CSV."""
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        from scipy.stats import pearsonr

        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found. Available columns: {self.df.columns.tolist()}"}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found. Available columns: {self.df.columns.tolist()}"}

        r, p_value = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p_value, 4)
        }

print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation between two columns in the loaded CSV. Returns correlation coefficient and p-value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name for correlation analysis."
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name for correlation analysis."
                    }
                },
                "required": ["col1", "col2"]
            },
        },
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one ReAct-agent loop using a simple tool-using agent.

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool.

    ACT:
      - If tools are requested, run the Python functions.

    OBSERVE:
      - Append each tool result back into the conversation history.
      - On the next iteration, the model reads those results and decides next step.

    Stop condition:
      - No tool calls in the response means it is the final answer.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                    print(f"RESULT: {result}")
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."

# =============================================================================
# Lesson 04: smolagents
# =============================================================================

# Q7

@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
    """
    Compute Pearson correlation between two columns in the loaded CSV.

    Args:
        col1: First column name for correlation analysis
        col2: Second column name for correlation analysis

    Returns:
        Dictionary with keys: col1, col2, pearson_r, p_value
    """
    return csv_backend.compute_correlation(col1, col2)

# Q8

@tool
def list_csv_files_tool() -> dict:
    """List available CSV files in resources/."""
    return csv_backend.list_csv_files()

@tool
def load_csv_tool(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/, e.g. 'bike_commute.csv'
    """
    return csv_backend.load_csv(filename)

@tool
def get_columns_tool() -> list:
    """Get the column names of the currently loaded CSV."""
    return csv_backend.get_columns()

@tool
def summarize_columns_tool(columns: list[str] | None = None) -> dict:
    """Show basic summary statistics for columns (uses pandas.describe).

    Args:
        columns: Optional list of column names. If omitted, summarize all columns.
    """
    return csv_backend.summarize_columns(columns)

@tool
def describe_column_tool(column: str) -> dict:
    """Show basic summary statistics for a single column.

    Args:
        column: Column name to describe.
    """
    return csv_backend.describe_column(column)

@tool
def plot_data_tool(y: str, x: str | None = None, plot_type: str = "line") -> str:
    """Plot data from the active CSV. If only y is provided, plot y vs row index.

    Args:
        y: Column name for y-axis
        x: Optional column name for x-axis
        plot_type: 'line' or 'scatter'
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)

TOOLS = [
    list_csv_files_tool,
    load_csv_tool,
    get_columns_tool,
    summarize_columns_tool,
    describe_column_tool,
    plot_data_tool,
    compute_correlation_tool,
]

# Configure the model
model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id="gpt-4o-mini",
)

# ToolCallingAgent
tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    max_steps=6,
)

# CodeAgent
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# =============================================================================
# Main
# =============================================================================

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# --- Lesson 02 ---

# Q1
tools = get_schema()

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
#    Yes, a tool call will be triggered because celsius_to_fahrenheit is in the schema.
# 2. How many API calls will be made to answer this query?
#    2
answer_with_agent = run_agent('Convert 100 degrees Celsius to Fahrenheit')
print(answer_with_agent)
# Was your prediction correct?
# Yes, the tool call was triggered as expected and two API calls were made.
# However, since run_agent only dispatches get_current_time at this point,
# celsius_to_fahrenheit was never actually executed as a Python function.
# The model answered correctly using its internal knowledge instead.

# Q3
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# The agent used a tool call (celsius_to_fahrenheit) because the user asked for a specific
# temperature conversion. Two API calls were made: one to request the tool, and another
# to generate the final answer after receiving the tool result.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# The agent did not use a tool call because the question asked for a conceptual explanation,
# not a numerical conversion. Only one API call was made, as the model answered directly
# using its internal knowledge.

# --- Lesson 03 ---

# Q4
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
answer = run_agent_cycle(
    messages,
    "Load bike_commute.csv. Then compute the correlation between distance_km and duration_min."
)
print("Answer:", answer)

# Q5
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print("Final Answer:")
print(result)

# Q6
# Roles in the ReAct loop:
# - "system": permanent instructions for the agent (initial context)
# - "user": the question or instruction from the user
# - "assistant": the model's response (may include tool_calls if tools are needed)
# - "tool": the result of executing a tool requested by the assistant
# This history shows the full sequence:
# system → user → assistant (with tool_calls) → tool → assistant (final response)
print("\n" + "=" * 60)
print("Full Message History (ReAct loop)")
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04 ---

# Q7
# Comparison between manual JSON schema and smolagents @tool decorator:
#
# INFORMATION SMOLAGENTS NEEDS FROM THE DEVELOPER:
# 1. The function name (inferred automatically)
# 2. The docstring with description and arguments (Google/NumPy format)
# 3. Type hints (col1: str, col2: str, -> dict)
#
# WHAT SMOLAGENTS GENERATES AUTOMATICALLY:
# 1. The complete JSON schema (name, description, parameters)
# 2. The description object with formatted documentation
# 3. The mapping between arguments and types
print(compute_correlation_tool.description)

# Q8
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n" + "=" * 60)
print("TOOLCALLING AGENT")
response_tool = tool_agent.run(prompt)
print("Response:", response_tool)

print("\n" + "=" * 60)
print("CODE AGENT")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})
print("Response:", response_code)

# Q8 Observations:
# 1. What did each agent produce?
#    Both agents loaded the CSV and created the scatter plot correctly.
#    However, neither agent changed the dot color to green.
#    ToolCallingAgent claimed in its final answer that it plotted "with green dots" — this is
#    a hallucination, since plot_data_tool has no color parameter and the dots are the default color.
#    CodeAgent used the plot_data_tool instead of writing custom matplotlib code,
#    so it also did not change the color.
#
# 2. What does this reveal about when each type of agent is more useful?
#    ToolCallingAgent is better for straightforward tasks covered by available tools,
#    but it cannot handle customization beyond what the tools support — and may hallucinate success.
#    CodeAgent should have written custom matplotlib code for the color styling,
#    but in this case it defaulted to using the tool instead, showing it does not always
#    generate code when it should. For styling tasks, CodeAgent with a clear system prompt
#    instructing it to write code for unsupported features would be the better choice.

# Q9
# 1. Task where ToolCallingAgent is better than CodeAgent:
#
#    Task: "Load bike_commute.csv and display the column names."
#
#    Why ToolCallingAgent is better:
#    - This task requires calling a single, well-defined tool (get_columns)
#    - No need for custom code generation
#    - ToolCallingAgent will directly call get_columns_tool and return the result
#    - CodeAgent might overcomplicate by writing unnecessary pandas code
#    - Tool-based approach is faster, cheaper (fewer tokens), and more predictable
#
#    Property that makes it a good fit for a tool-based approach:
#    - The task matches exactly one of the available tools' capabilities
#    - No customization or creative problem-solving is required
#    - The inputs and outputs are clearly defined and predictable
#
# 2. One meaningful risk of using CodeAgent that does not apply to ToolCallingAgent:
#
#    Risk: Execution of arbitrary generated code that could be unsafe or destructive.
#
#    Explanation:
#    - CodeAgent writes and executes Python code on your system
#    - If the agent generates malicious or buggy code, it could:
#      * Delete or corrupt files
#      * Access sensitive information
#      * Create infinite loops that crash the system
#      * Import dangerous modules
#    - ToolCallingAgent only calls pre-approved Python functions you wrote
#    - ToolCallingAgent cannot execute arbitrary code or access system resources
#    - This is why smolagents implements sandboxed execution for CodeAgent
#
#    Real-world example:
#    - A CodeAgent asked to "clean up disk space" might generate:
#      import os; os.system("rm -rf /") (theoretically dangerous)
#    - A ToolCallingAgent cannot do this because no such tool exists
