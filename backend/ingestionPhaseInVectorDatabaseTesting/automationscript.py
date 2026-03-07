from pymilvus import MilvusClient, DataType
from mainoptimized import main

uri = "https://in03-595f70b7f6c6a77.serverless.gcp-us-west1.cloud.zilliz.com"
user = "db_595f70b7f6c6a77"
password = "Ef6/cx*r6{CWP,,M"
client = MilvusClient(uri = uri, user = user, password = password, timeout = 10000)
try :
    response = client.delete(
        collection_name="stresstestingragingestionphase",
        filter = "id >= 0"
    )
    client.close()
except Exception as e:
    print(f"An error occurred while deleting data: {e}")
    client.close()


























#................................................................
# results = client.query(
#     collection_name="childrenEmbeddingLateChunking",
#     filter="uniqueIdForEmbedding >= 0",
#     output_fields=["uniqueIdForEmbedding"]
# )

# results1 = client.query(
#     collection_name="collectTextWithMetadata",
#     filter="uniqueIdForText >= 0",
#     output_fields=["uniqueIdForText"]
# )
# # print(results)
# # print(type(results))
# unique_ids = {row["uniqueIdForEmbedding"] for row in results}
# unique_ids1 = {row["uniqueIdForText"] for row in results1}

# # unique_ids1 = {print(row.keys) for row in results}
# # print(unique_ids1)
# print(unique_ids)
# print(unique_ids1)
# for id in unique_ids:
#     if id not in unique_ids1:
#         print(id)
# print(len(unique_ids1))
# print(len(unique_ids))
# if len(unique_ids) == len(unique_ids1) :
#     print("same")
# # print(type(unique_ids))
# # count = len(unique_ids)
# # listIds = list(unique_ids)
# # print(max(listIds))
# # print("Total unique uniqueIdForEmbedding:", count)
# # print(unique_ids)

# print(client)
#..............................................................