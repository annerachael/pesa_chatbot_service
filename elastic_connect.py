import csv
import warnings
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import os

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="elasticsearch")

# Connect to Elasticsearch Cloud
client = Elasticsearch(
    "https://pesa-chatbot-analysis.es.eu-central-1.aws.cloud.es.io",
    verify_certs=False,
    basic_auth=("elastic", os.environ.get("ELASTIC_PASSWORD")),
)

def bulk_upload_csv_to_elasticsearch(file_path, index_name):
    try:
        # Open the CSV file
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)  # Automatically uses the first row as field names
            actions = []

            # Convert each row in CSV to Elasticsearch bulk action format
            for row in reader:
                action = {
                    "_index": index_name,
                    "_source": row,  # The row itself becomes the document
                }
                actions.append(action)

            helpers.bulk(client, actions)
            print(f"Successfully indexed data into '{index_name}' index.")
    except Exception as e:
        print(f"Error: {e}")


csv_file_path = "McDonalds_Financial_Statements.csv"
index_name = "pesa_chatbot_analysis"

bulk_upload_csv_to_elasticsearch(csv_file_path, index_name)
