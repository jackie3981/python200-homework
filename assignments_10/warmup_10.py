# --- LLMs as Transform ---
# LLMs as Transform Question 1

# For each task below, write a one-sentence comment saying whether you would use an LLM or deterministic code, and why.

# - Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
#   Deterministic code, because date parsing follows strict, predictable rules and libraries like datetime can handle this.
# - Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
#   LLM, because it is the most common use case, natural language variations in describing issues make rule-based classification brittle. 
#   These require reading comprehension and judgment.
# - Calculate the average of a list of numbers.
#   Deterministic code, as it is a simple mathematical operation with fixed logic.
# - Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
#   LLM, because the position of "@" or indicators like "at" can vary, and company names themselves are arbitrary strings. Code-based approaches require to anticipate every variant in advance.
# - Determine whether a product review is more than 100 words long.
#   Deterministic code — token splitting on whitespace or using a simple character/word count function is perfectly accurate and more efficient than using an LLM.

# LLMs as Transform Question 2
# PROBLEM: The output is unstructured natural language of variable length and format, difficult to parse in a pipeline or validate 
# the result because there are no constraints on length, sentence count, or format — leading to fragility, storage inefficiency, 
# and unpredictable pipeline behavior.
# system1 = "Summarize this product review in exactly one sentence, no more than 20 words. Output only the summary, with no extra text, explanations, or quotation marks."
# Another system prompt if classification is needed over summarize
# system2 = "Classify the sentiment of this product review. Reply with exactly one word: positive, negative, or neutral."

# LLMs as Transform Question 3
# If each call takes 1 second, sequential processing of 50,000 records would take:
# 50,000 seconds (aprox: 833 minutes, aprox: 14 hours) is too slow for most production pipelines
# Practical strategy for scale (without changing models):
# Use OpenAI's Batch API, which processes requests asynchronously in bulk.
# The Batch API typically offers lower cost and does not require real-time parallelism management, rate-limit handling, or custom async code.

# --- Azure OpenAI ---
# Azure OpenAI Question 1
# - In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.
# Two specific reasons an organization might use Azure OpenAI instead of the OpenAI API:
# 1. Data residency and compliance — requests stay inside Azure's infrastructure, which is required for regulated industries like healthcare, finance, and government.
# 2. Unified billing and support — Azure OpenAI costs appear on the same bill as other Azure services, and support goes through Microsoft rather than
#    managing a separate vendor relationship with OpenAI.

# Azure OpenAI Question 2
# - When you switch from OpenAI to AzureOpenAI, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard api_key -- describe the Azure-specific ones.)
# The three Azure-specific parameters when switching from OpenAI to AzureOpenAI are:
# 
# 1. azure_endpoint — The URL of your Azure OpenAI resource (format: "https://{your-resource}.openai.azure.com/")
#    This is required because requests must route through Azure's infrastructure, not OpenAI's direct API.
# 
# 2. azure_deployment — The name of the specific model deployment you created in your Azure OpenAI resource (e.g., "gpt-4o-mini" or "gpt-5.4"). 
#    Azure requires this because you can have multiple deployments of the same model with different configurations. 
# 
# 3. api_version — The API version string (e.g., "2024-02-15-preview" or "2025-01-01-preview") Azure versions its OpenAI API independently from OpenAI. 

# Azure OpenAI Question 3
# - In a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". 
# What does it take instead, and where do you find the right value to use?
# Using Azure OpenAI (enterprise / production approach)
# from openai import AzureOpenAI

# client = AzureOpenAI(
#     azure_endpoint="https://<resource-name>.openai.azure.com",
#     api_key="<azure-api-key>",
#     api_version="2024-02-01"
# )
# "model" takes a deployment name, not a model name. In Azure OpenAI, you do not call a model directly, you call a named deployment 
# that your organization's admin created and configured.

# Where to find the correct deployment name:
# 1. Navigate to Azure AI Foundry (https://ai.azure.com)
# 2. Open your Azure OpenAI resource
# 3. Go to the "Deployments" section — the deployment names are listed in the first column
#