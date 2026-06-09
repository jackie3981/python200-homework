# https://youtu.be/pcAYPYZxtNI

# Step 6: Reflect
# - Was classifying weather conditions for outdoor running actually a good use of an LLM? 
# No, it is not a good use, this could be solved using rules like "if". It is simple, faster, easy to modify, and free.
# - Could deterministic code have done this better? What would you lose or gain by switching to a rule-based approach (e.g., "temperature > 10 and precipitation < 1 → good")?
# Gain by switching to rules: cost savings, consistency, speed, offline capability
# Lose by switching to rules: contextual interpretation, flexibility with new conditions

import json, os
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError

BASE_DIR = os.path.dirname(__file__)
output_path = os.path.join(BASE_DIR, "outputs")

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"
processed_path = f"processed/{today}/weather_classified.json"

load_dotenv()

ACCOUNT_URL = "https://yoicelctd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"
VALID_LABELS = {"good", "marginal", "bad"}
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

# Step 1: Read
def authentication():
    '''Authenticate to Azure Blob Storage using DefaultAzureCredential and return a ContainerClient'''
    credential = DefaultAzureCredential()
    container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
    return container

def load_fallback_data():
    '''Load fallback weather data from a previous assignment if today's blob is not found'''
    fallback_path = os.path.join(os.path.dirname(BASE_DIR), 'assignments_09/outputs/weather_raw.json')  
    with open(fallback_path, 'r') as f:
        data = json.load(f)
    return data

def download_data(container):
    '''Download the raw weather data for today from Azure Blob Storage. If it doesn't exist, load fallback data from a previous assignment.'''
    try:
        # Try to download from blob storage
        raw = container.download_blob(blob_path).readall()
        data = json.loads(raw.decode("utf-8"))
        print(f"Loaded data from blob: {blob_path}")
    except ResourceNotFoundError:
        print(f"Blob not found for {today}, using fallback dataset")
        data = load_fallback_data()

    hourly = data["hourly"]
    records = []
    for i in range(len(hourly["time"])):
        record = {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        records.append(record)

    print(f"Loaded {len(records)} hourly records")
    return records

# Step 2: Transform
def make_user_message(record):
    '''Format the user message for the LLM based on the weather record'''
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

def classify_weather(records):
    '''
    Use OpenAI GPT-4o-mini to classify weather conditions for the first 24 records.
    Returns a list of enriched records with a "conditions" key added.
    '''
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    enriched = []
    for i, record in enumerate(records[:24]):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(record)},
            ]
        )

        raw_label = (response.choices[0].message.content or "unknown").strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"  Processed {i + 1} records...")
  
    return enriched

# Step 3: Write
def upload_data(container, enriched):
    '''Upload the enriched records back to Azure Blob Storage as a JSON file'''
    container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
    print(f"Uploaded to {processed_path}")

# Step 4: Spot-Check
def data_check(container):
    '''
    Download raw weather data for today from Azure Blob Storage and reshape
    the hourly parallel lists into a list of per-hour record dictionaries.
    Falls back to a local file if today's blob is not found.
    '''
    processed_data = container.download_blob(processed_path).readall()
    enriched = json.loads(processed_data.decode("utf-8"))

    df = pd.DataFrame(enriched)
    print("\nLabel distribution:")
    print(df["conditions"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())

    return enriched

# Step 5: Save Output
def save_output(enriched):
    '''Save the first 10 enriched records to a local JSON file for easy access'''
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "first_10_records.json")
    with open(output_file, "w") as f:
        json.dump(enriched[:10], f, indent=2)  # Save only first 10
    print(f"\nSaved first 10 records to {output_file}")



if __name__ == "__main__":
    print(f"Processing date: {today}")
  
    # Step 1: Read
    print("\n[Step 1] Reading weather data...")
    container = authentication()
    records = download_data(container)

    # Step 2: Transform
    print("\n[Step 2] Classifying weather conditions with OpenAI...")
    enriched_records = classify_weather(records)

    # Step 3: Write
    print("\n[Step 3] Uploading classified data...")
    upload_data(container, enriched_records)

    # Step 4: Spot-Check
    print("\n[Step 4] Running spot check...")
    final_data = data_check(container)

    # Step 5: Save Output
    print("\n[Step 5] Saving first 10 records...")
    save_output(final_data)



    
