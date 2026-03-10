from pymilvus import MilvusClient
URI = "https://in03-46f4508aa296b4c.serverless.gcp-us-west1.cloud.zilliz.com"
USER = "db_46f4508aa296b4c"
PASSWORD = "Or9.c{8g1[z}gym]"
TOKEN = "883506b5ce7154d0c7c34c19f3ba8412d6fba41e629333a04e17e3aa37f2f0654818fe3f6d2ed6671b5a2dad05d9a1b16510329a"
# client = MilvusClient(uri = URI, user = USER, password = PASSWORD, timeout = 10000)
client = MilvusClient(uri = URI, token = TOKEN)
print(client)
print(client.list_collections())
client.close()