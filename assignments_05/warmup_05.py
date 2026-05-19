import json
from dotenv import load_dotenv
from openai import OpenAI

# *** The Chat Completions API ***
# API Q1 
# API Q2: adding temperature. default set to 0.
# API Q3: adding n, default set to 1.
# API Q4: adding max_tokens, default set to 100.
def question_answer(question, temp=0, n=1, tokens=100):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature= temp,
        n=n,
        max_tokens=tokens
    )
    return response

def answers_api_questions():
    # API Question 1: Use the model "gpt-4o-mini" and send this prompt: "What is one thing that makes Python a good language for beginners?". 
    # Print the model's response, the name of the model, and the total number of tokens used.
    question = "What is one thing that makes Python a good language for beginners?"
    answer = question_answer(question=question)
    print(f"\nAnswer: {answer.choices[0].message.content}")
    print(f"\nModel used: {answer.model}")
    print(f"\nTokens used: {answer.usage.total_tokens}")

    # API Question 2: Using the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.
    prompt = "Suggest a creative name for a data engineering consultancy."
    temperatures = [0, 0.7, 1.5]
    for i in temperatures:
        name_sugested = question_answer(question=prompt, temp=i)
        print(f"\nName sugested with temperature {i}: {name_sugested.choices[0].message.content}")
    # *********************************************************************************************************************************************************************
    # The lowest temperature value (close to 0) will produce a more consistent response. As the value moves away from 0, the responses become more random and creative.
    # For consistency output, temperature = 0 should be used. 
    # *********************************************************************************************************************************************************************

    # API Question 3: Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three.
    temperature=1.0
    n=3
    prompt = "Give me a one-sentence fun fact about pandas (the animal, not the library)."
    fun_fact_pandas = question_answer(question=prompt, temp=temperature, n=n)
    for i in range(n):
        print(f"\nFun fact about pandas {i+1}: {fun_fact_pandas.choices[i].message.content}")

    # API Question 4: Set max_tokens=15 and send a prompt that would normally produce a long response. 
    # Print the result. Add a comment: What happened, and why might you want to use max_tokens in a real application?
    max_tokens=15
    prompt = "Explain how neural networks work."
    answer_maxtokens = question_answer(question=prompt, tokens=max_tokens)
    print(f"\nAnswer with limited tokens: {answer_maxtokens.choices[0].message.content}")
    # *******************************************************************************************************************************************************
    # By using 'max_tokens', we are limiting the AI ​​to a certain number of tokens(aprox 0.75 word per token), which causes the response to be cut short, thereby affecting its quality. 
    # It is appropriate to use 'max_tokens' when you want to generate short responses, display messages that fit your UI, or save on costs—though this does 
    # not guarantee that the resulting response will be complete.
    # *******************************************************************************************************************************************************

# *** System Messages and Personas ***
# System Q1
def question_answer_personality(system_content, user_question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": user_question}],
    )
    return response

# System Q2
def conversation_memory(msg):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg
    )
    return response

def answer_system_questions():
    # System message Q1: Use a system message to give the model a personality, then ask it a question. Print the response.
    # Change the system message to give the model a completely different personality (your choice) and ask the same question. Print that response.
    system_content1 = "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    system_content2 = "You give short answers using as few words as possible."
    user_question = "I don't understand what a list comprehension is."

    answer1 = question_answer_personality(system_content1, user_question)
    print(f"\nAnswer with personality 1: {answer1.choices[0].message.content}")
    answer2 = question_answer_personality(system_content2, user_question)
    print(f"\nAnswer with personality 2: {answer2.choices[0].message.content}")
    # ****************************************************************************************************
    # The first model gives a more detailed answer, while the second one gives a short and concise answer. 
    # This happens because the system message control the behavior and the personality of the model.
    # ****************************************************************************************************

    # System message Q2: Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call. 
    # Print the model's response. Add a comment: Why does the model know Jordan's name, even though it's stateless?
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
        {"role": "user", "content": "Can you remind me what my name is?"}
    ]
    answer = conversation_memory(messages)
    print(f"\nConversation with memory: {answer.choices[0].message.content}")
    # *********************************************************************************************
    # The model remembers the name because the conversation history is included in the messages.
    # *********************************************************************************************

# *** Prompt Engineering ***
def classification_prompt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response

# Prompt Q1
def prompt_q1(reviews, base_prompt):
    for i, review in enumerate(reviews,start=1):
        prompt = f"""
        {base_prompt}
        Review: {review}
        """
        answer = classification_prompt(prompt)
        print(f"\nReview {i}: {answer.choices[0].message.content}")

# Prompt Q2
def prompt_q2(reviews, base_prompt, example):
    for i, review in enumerate(reviews,start=1):
        prompt = f"""
        {base_prompt}
        {example}
        Review: {review}
        """
        answer = classification_prompt(prompt)
        print(f"\nReview {i}: {answer.choices[0].message.content}")
    # ****************************************************************************************************************
    # Adding an example helps the model to understand the task and provides a clear format for the expected output.
    # ****************************************************************************************************************
    
# Prompt Q3
def prompt_q3(reviews, base_prompt, example):
    for i, review in enumerate(reviews,start=1):
        prompt = f"""
        {base_prompt}
        {example}
        Review: {review}
        """
        answer = classification_prompt(prompt)
        print(f"\nReview {i}: {answer.choices[0].message.content}")
    # *************************************************************************************************************************************************
    # zero-shot: when there is no example available, it's faster and requires less tokens.
    # one-shot: when you want to guide the model with less tokens, it gives the model a basic guidance.
    # few-shot: when you need consistency and accuracy, it provides more examples and gives the most reliable output, it requires more tokens.
    # *************************************************************************************************************************************************

# Prompt Q5
def prompt_structure_q5(base_prompt):
    answer_q5 = classification_prompt(base_prompt)
    raw_output = answer_q5.choices[0].message.content
    print(f"\nRaw Output:\n {raw_output}")
    try:
        data = json.loads(raw_output)
        print("\nParsed output:")
        print(f"Sentiment: {data['sentiment']}")
        print(f"Confidence: {data['confidence']}")
        print(f"Reason: {data['reason']}")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        print(f"Raw output: \n {raw_output}")

# Prompt Q6
def prompt_delimiters(base_prompt):
    answer = classification_prompt(base_prompt)
    print(f"\nAnswer: \n {answer.choices[0].message.content}")

def answer_prompt_questions():
    reviews = [
        "The onboarding process was smooth and the team was welcoming.",
        "The software crashes constantly and support never responds.",
        "Great price, but the documentation is nearly impossible to follow."
    ]

    review_q5 = "I've been using this tool for three months. It handles large datasets well, \
          but the UI is clunky and the export options are limited."

    base_prompt = "Classify the following reviews as positive, negative, or mixed. "

    example_q2 = """
    Example:
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed
    """

    example_q3 = """
    Examples:
    Review: "I like to play sports, it help me to stay healthy and in shape."
    Sentiment: Positive

    Review: "This is the worst experience I've had. Nothing works and it's frustrating."
    Sentiment: Negative

    Review: "The service is okay, the personnel is friendly, but the wait is too long."
    Sentiment: Mixed
    """

    prompt_q4 = """
    Solve the following problem step by step before giving the final answer.

    A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.
    What is her final annual salary?

    Show your reasoning step by step and clearly label the final answer.
    """ 

    prompt_q5 = f"""
    analyze the review and return the result only as valid JSON.
    Keys: sentiment, confidence (float from 0 to 1), and reason (one sentence).

    Review: {review_q5}
    """

    user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
          pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
    user_text2 = "I came here tonight because when you realize you want to spend the rest of your life with somebody, you want the rest of your life to start as soon as possible."

    prompt_q6_instruction = f"""
    You will be given text inside triple backticks.
    If it contains step-by-step instructions, rewrite them as a numbered list.
    If it does not contain instructions, respond with exactly: "No steps provided."

    ```{user_text}```
    """

    prompt_q6_no_instruction = f"""
    You will be given text inside triple backticks.
    If it contains step-by-step instructions, rewrite them as a numbered list.
    If it does not contain instructions, respond with exactly: "No steps provided."

    ```{user_text2}```
    """

    # Prompt Q1 — Zero-Shot: Ask the model to classify the sentiment of each review below as positive, negative, or mixed. 
    # Give it no examples — just the task description and the reviews. Print each result labeled with the review number.
    prompt_q1(reviews, base_prompt)
    
    # Prompt Q2 — One-Shot: Repeat the same task, but this time add one example before the reviews to show the model the format you want. 
    # Print the results. Add a comment: Did adding one example change the format or consistency of the output compared to Q1?
    prompt_q2(reviews, base_prompt, example_q2)

    # Prompt Q3 — Few-Shot: Repeat the task again, this time with three examples. At least one example should be positive, one negative, and one mixed. 
    # Print the results. Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
    prompt_q3(reviews, base_prompt, example_q3)

    # Prompt Q4 — Chain of Thought: Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. 
    # Label the final answer clearly. Print the full response including the reasoning. Add a comment: Why does asking the model to reason step by step 
    # tend to improve accuracy on problems like this?
    answer_q4 = classification_prompt(prompt_q4)
    print(f"\nSolution step by step:\n {answer_q4.choices[0].message.content}")
    # *******************************************************************************************************************************************
    # step by step improves the accuracy because it forces the model to break the problem into smaller parts, which helps to avoid mistakes.
    # *******************************************************************************************************************************************
    
    # Prompt Q5 — Structured Output: Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment, 
    # confidence (a float from 0 to 1), and reason (one sentence). Print the raw response, then parse it with json.loads() and print each field separately, labeled.
    prompt_structure_q5(prompt_q5)

    # Prompt Question 6 — Delimiters: Use triple backticks as delimiters to clearly separate the user's text from your instructions. Send the prompt below and print the result.
    # Then send a second prompt using a passage that is not a set of instructions (any sentence or two of regular prose). Confirm that the model returns "No steps provided." 
    # Add a comment: What problem do delimiters help prevent?
    prompt_delimiters(prompt_q6_instruction)
    prompt_delimiters(prompt_q6_no_instruction)
    # ****************************************************************
    # Delimiters helps  to prevent the model to confuse the instructions inside the user input with the actual system instructions, improving reliability.
    # ****************************************************************

# *** Local Models with Ollama ***
# Ollama Q1
ollama_answer = """
A large language model is an AI system designed to understand and generate human language, enabling it to process
and respond to user input in natural ways. It is trained on vast amounts of text, allowing it to learn context,
understand nuances, and generate coherent responses.

**Example**: A large language model can understand a user's query, identify relevant context, and produce a
meaningful response, even when the input is ambiguous or complex.
"""

def ollama_q1():
    prompt = "Explain what a large language model is in two sentences."
    answer = classification_prompt(prompt)
    print(f"\nAnswer: {answer.choices[0].message.content}")
    # ***************************************************************
    # OpenAI answer is more technical and concise, while Ollama answer is more conceptual and includes an example. 
    # Advantage of running a model locally: no API costs and no data is sent to external servers, which improves privacy.
    # Disadvantage: local models are generally smaller and less capable than cloud-based models like GPT-4o-mini.
    # ****************************************************************

# -- Main --
if load_dotenv():
    print("Successfully loaded api key")
    client = OpenAI()

    #answers_api_questions()
    #answer_system_questions()
    #answer_prompt_questions()
    ollama_q1()

else:
    print("Fail to load api key")