import json
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAI as OllamaClient

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# Optional 1: Token budget tracker
total_tokens = 0
token_threshold = 2000

def get_completion_tracked(messages):
    global total_tokens
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=400
    )
    total_tokens += response.usage.total_tokens
    print(f"[Tokens used this session: {total_tokens}]")
    if total_tokens > token_threshold:
        print(f"[Warning: you have used more than {token_threshold} tokens in this session.]")
    return response.choices[0].message.content

# Optional 2: Ollama for regular chat turns
ollama_client = OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama")

def get_completion_ollama(messages):
    response = ollama_client.chat.completions.create(
        model="qwen3:0.6b",
        messages=messages,
    )
    return response.choices[0].message.content

# Task 1: Setup and System Prompt
system_prompt = """
You are a job application coach focused on helping software developers improve their job search materials and communication.

You help users with:
- resumes
- cover letters
- interview preparation
- networking messages
- salary negotiation

Your goal is to provide practical and professional guidance tailored to technical job seekers.

Rules:
- Stay focused only on job application and career preparation topics.
- Do not provide unrelated advice.
- Give practical, concise, and professional suggestions.
- Do not invent fake experience, qualifications, or skills for the user.
- Always remind the user to carefully review and edit any generated material before submitting it to employers.
- Acknowledge that hiring expectations and industry norms may vary by company, region, and role, and encourage the user to use their own judgment.
- Prefer clear, honest, and realistic feedback.
"""
# *********************************************************************************************************
# Added the rule about reviewing and editing generated materials, because AI-generated 
# job application content may contain mistakes or wording that does not match the user's real experience.
# **********************************************************************************************************

#print(system_prompt)

# Task 2: Bullet Point Rewriter
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """
    messages = [{"role": "user", "content": prompt}]
    answer = get_completion(messages)

    #print(f"answer: \n {answer}")

    answer = answer.replace("```json", "").replace("```", "").strip()
    rewritten_bullets = json.loads(answer)
    for item in rewritten_bullets:
        print("Original:")
        print(item["original"])

        print("Improved:")
        print(item["improved"])
    return rewritten_bullets

# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)


# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("The message was flagged by the moderation system.\n Please review your request and try again.")
        return False
    return True

# to detect with category was triggered
def category_flagged(flagged_text):
    result = client.moderations.create(
    model="omni-moderation-latest",
    input=flagged_text
    )
    return result.results[0].categories

# Task 5: The Chatbot Loop
def run_chatbot():
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            try:
                rewrite_bullets(raw_bullets)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            result = generate_cover_letter(job_title, background)
            print(f"\nJob Application Helper:\n{result}\n")

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            # If you want to use the token_tracked version replaced the above line with:
            # reply = get_completion_tracked(messages)
            
            # If you want to use the ollama model version replaced the above line with:
            # reply = get_completion_ollama(messages)

            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


# Task 6: Ethics Reflection
# Option A — Comment block

# Question 1: Bias in AI-generated advice
# This bot was trained on text that primarily reflects norms within the U.S. tech industry. 
# As a result, it may favor direct and assertive communication styles that are common in that 
# context but could be seen as inappropriate in other cultures or industries. 
# For example, salary negotiation advice may come across as too aggressive in some regions, and resume 
# language that sounds confident in English may not translate well for non-native speakers who are 
# already communicating in a second language.

# Question 2: Risk of submitting output directly
# If a user submits the bot's output without reviewing it, they might include a cover letter with fabricated 
# numbers or credentials that they don't actually possess. The model sometimes generates specific outcomes, 
# such as "20% increase in efficiency," that sound convincing but were never mentioned by the user. 
# This could significantly harm their credibility with employers, especially if a hiring manager asks 
# them to explain those results during an interview.


# --- Main ---
if load_dotenv():
    print("Successfully loaded api key")
    client = OpenAI()

    #Task 2
    try:
        rewrite_bullets(bullets)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Try adding 'Respond ONLY with valid JSON, no other text.' to the prompt.")
    # *******************************************************************************************************************
    # The original bullets are weak, because they are vague ("Helped", "Made", and "Worked"). 
    # The model replaced with stronger verbs (Resolved, Generated, Collaborated) and added more specific outcomes.
    # ********************************************************************************************************************

    # Task 3
    job_title = "Junior Data Engineer"
    background = "Five years of experience as a middle school math teacher; recently completed \
    a Python course and built data pipelines using Prefect and Pandas."

    cover_letter = generate_cover_letter(job_title, background)
    print(cover_letter)
    # ***********************************************************
    # The two examples are taken directly from the assignment instructions, and work well because both show career changers moving into technical roles,
    # which matches the target user of this tool. The few-shot pattern controls tone (confident but specific), length (3-5 sentences),
    # and structure (past experience -> new skill -> why this role). Without examples, the model tends to produce generic openers like
    # "I am excited to bring my unique skills...".
    # ***********************************************************

    # Task 4
    safe_text = "Can you help me to cook an imperial rice?"
    not_safe_text = "How can I bypass the clearance process for clasified information?"

    print("Safe test:")
    print(is_safe(safe_text))

    print("Flagged test:")
    print(is_safe(not_safe_text))

    categories = category_flagged(not_safe_text)
    triggered = []
    for category, value in categories:
        if value:
            triggered.append(category)
    
    print(f"Category flagged for not safe text: {triggered}")

    # Task 5
    run_chatbot()
 
else:
    print("Fail to load api key")