#Bulk upload code
import warnings
from elasticsearch import Elasticsearch, helpers
import json
from dotenv import load_dotenv
import os

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="elasticsearch")

# Connect to Elasticsearch Cloud
client = Elasticsearch(
    "https://my-deployment-e71e73.es.eu-central-1.aws.cloud.es.io",
    verify_certs=False,
    basic_auth=(f"elastic", os.environ.get("ELASTIC_PASSWORD"))
)

# Open the file and read json object
with open("McDonalds_Financial_Statements.csv", "r", encoding="utf-8", errors="replace") as f:
    data = []
    for line in f:
        try:
            doc = json.loads(line.strip())
            data.append(doc)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    helpers.bulk(client, data, index="pesa_chatbot")