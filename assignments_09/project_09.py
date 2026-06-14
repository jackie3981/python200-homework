# https://youtu.be/bRMn0hymSNE

import requests, json, os
import pandas as pd
from datetime import date
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

BASE_DIR = os.path.dirname(__file__)
output_path = os.path.join(BASE_DIR, "outputs")

ACCOUNT_URL = "https://yoicelctd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"
# Charlotte, NC is latitude=35.2271&longitude=-80.8431 by default.
LATITUDE = 35.2271
LONGITUDE = -80.8431

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

# Step 1: Extract
def extract_REST_API(lat=LATITUDE, lon=LONGITUDE):
    '''
    Call the Open-Meteo API to retrieve 7 days of hourly weather data.
    '''
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data

# Step 2: Serialize
def serialize_json(data):
    '''
    Convert the API response to JSON bytes using json.dumps() and .encode("utf-8").
    '''
    payload = json.dumps(data).encode("utf-8")
    return payload

# Step 3: Load
def upload_to_blob(payload):
    '''
    Uploads the serialized data to Blob Storage.
    '''
    credential = DefaultAzureCredential()
    container = ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER,
        credential=credential
    )

    container.upload_blob(blob_path, payload, overwrite=True)
    print(f"Uploaded to {blob_path}")
    return container

# Step 4: Verify
def list_all_blob(container):
    '''
    List all blobs in the container and print each one's name and size.
    '''
    for blob in container.list_blobs():
        print(f"  - {blob.name} ({blob.size} bytes)")

# Step 5: Read Back
def download_blob(container):
    '''
    Downloads the blob, prints first 5 rows of the DataFrame, and saves the raw JSON to outputs/weather_raw.json
    '''
    raw = container.download_blob(blob_path).readall()
    data_back = json.loads(raw.decode("utf-8"))

    df = pd.DataFrame(data_back["hourly"])
    print(df.head())

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "weather_raw.json"), "w") as f:
        json.dump(data_back, f, indent=2)
    
    return data_back

if __name__ == "__main__":
     # Step 1
    data = extract_REST_API()

    # Step 2
    payload = serialize_json(data)
    
    # Step 3
    container = upload_to_blob(payload)
 
    # Step 4
    list_all_blob(container)
    
    # Step 5
    download_blob(container)

