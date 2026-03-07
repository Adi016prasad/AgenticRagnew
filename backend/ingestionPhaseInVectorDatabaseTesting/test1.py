from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
import os

load_dotenv()

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USERNAME")
MILVUS_PASS = os.getenv("MILVUSPASSWORD")

uri = MILVUS_URI
user = MILVUS_USER
password = MILVUS_PASS
client = MilvusClient(uri = MILVUS_URI, user = MILVUS_USER, password = MILVUS_PASS, timeout = 10000)

print(client.list_collections())
print(client.list_indexes(collection_name="stresstestingragingestionphase"))

client.close()