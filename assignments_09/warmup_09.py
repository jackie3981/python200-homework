from azure.identity import DefaultAzureCredential
from azure.mgmt.resource.subscriptions import SubscriptionClient

credential = DefaultAzureCredential()
client = SubscriptionClient(credential)

for sub in client.subscriptions.list():
    print(sub.display_name)

# --- Azure Authentication ---
# Azure Authentication Question 1
# In a comment block, answer: when you run a Python script locally that uses DefaultAzureCredential, 
# what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?
"""
When you run a Python script locally that uses DefaultAzureCredential, it relies on your existing 'az login' session.

The command you must run first is:
    az login

DefaultAzureCredential knows to use it because it tries a sequence of authentication methods in order, and one of them picks up your 'az login' 
session. That's why, after running 'az login', your script works without writing any additional authentication code.
"""

# Azure Authentication Question 2
# In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or container) 
# use az login for authentication? What does it use instead, and why does the same Python code work without changes?
"""
A deployed pipeline (running on an Azure VM or container) cannot use 'az login' because it requires a human to run it. 
In an automated pipeline, there is no one present to type the command or authenticate in the browser.

Instead, it uses Managed Identity. When a VM or container has a managed identity assigned to it, 
DefaultAzureCredential detects and uses it automatically.

The same Python code works without changes because DefaultAzureCredential tries multiple authentication methods in order. 
Locally it uses 'az login', and in the cloud it uses Managed Identity.
"""

# Azure Authentication Question 3
# You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. 
# In a comment block, describe the two most likely causes and how you would diagnose each.
"""
Two most likely causes of an AuthenticationError with DefaultAzureCredential:

CAUSE 1: You haven't run 'az login' recently
    - 'az login' sessions expire after a period of inactivity
    - To diagnose: run 'az login' again and test your script

CAUSE 2: Your account does not have permissions on the subscription
    - Your account may be authenticated but does not have access to resources
    - To diagnose: check in the Azure Portal that your user has the necessary roles on the subscription (like 'Reader')

If your script fails, first confirm that 'az login' works, then check your account permissions on the subscription.
"""

# Additional note
"""
ERROR:
    ImportError: cannot import name 'SubscriptionClient' from 'azure.mgmt.resource'

CAUSE:
    The structure of the 'azure-mgmt-resource' package changed in recent versions.
    SubscriptionClient can no longer be imported directly from 'azure.mgmt.resource'.
    It now resides in a specific submodule called 'subscriptions'.

DIAGNOSIS:
    1. The original code used:
       from azure.mgmt.resource import SubscriptionClient
    
    2. When executed, Python could not find 'SubscriptionClient' in that location
    
    3. Check the installed version:
       uv pip show azure-mgmt-resource

SOLUTION:
    Change the import statement to:
       from azure.mgmt.resource.subscriptions import SubscriptionClient

    Or alternatively, install the separate package:
       uv pip install azure-mgmt-resource-subscriptions

HOW I FIXED IT:
    I modified the import in my script from:
       from azure.mgmt.resource import SubscriptionClient
    to:
       from azure.mgmt.resource.subscriptions import SubscriptionClient
    
    After this change, the script was able to import the class correctly.
"""

# --- Blob Storage ---
# Blob Storage Question 1
# In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. 
# Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).
"""
The three-level hierarchy of Azure Blob Storage is:

1. STORAGE ACCOUNT - This is the top level. Think of it as the main resource that holds everything. 
Each storage account has a unique name and its own URL.

2. CONTAINER - This is inside the storage account. Think of it as a folder that groups related blobs together. 
You can have many containers in one account.

3. BLOB - This is the actual file. It can be anything - a CSV, JSON, image, etc. 
   You can use slashes in the blob name to make it look like it's inside subfolders.

CONCRETE ANALOGY (apartment building):
- STORAGE ACCOUNT = The entire apartment building
- CONTAINER = One apartment unit inside the building
- BLOB = A specific piece of furniture inside that apartment

ANOTHER ANALOGY (library):
- STORAGE ACCOUNT = The whole library building
- CONTAINER = One bookshelf in the library
- BLOB = A single book on that shelf
"""

# Blob Storage Question 2
# For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.
#  * A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
#    Use BLOB STORAGE. It needs to store the raw responses as complete files for reprocessing later, which is exactly what Blob Storage is designed for.

#  * Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
#    Use A RELATIONAL DATABASE (like Azure SQL). You need to query data by values (date range and customer ID), filter, and aggregate,
#    this is what databases are designed for, not Blob Storage.

#  * A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
#    Use BLOB STORAGE. It needs to store files as-is (NumPy arrays) between pipeline runs without querying individual records,
#    this is a perfect use case for Blob Storage.

# Blob Storage Question 3
# Write a function list_container(container_client) that prints the name and size (in bytes) of every blob in the container, 
# one per line. The function should take a ContainerClient object as its only argument and return nothing.
def list_container(container_client):
    """
    Prints the name and size (in bytes) of every blob in the container, one per line.
    
    Args:
        container_client: A ContainerClient object
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name} ({blob.size} bytes)")

# Blob Storage Question 4
# Write a function upload_text(container_client, blob_name, text) that encodes a Python string as UTF-8 and uploads it as a blob, 
# overwriting any existing blob with the same name. The function should take a ContainerClient, a blob name string, and a text string, and return nothing.
def upload_text(container_client, blob_name, text):
    """
    Encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob.
    
    Args:
        container_client: A ContainerClient object
        blob_name: The name/path for the blob
        text: The string text to upload
    """
    container_client.upload_blob(blob_name, text.encode("utf-8"), overwrite=True)
