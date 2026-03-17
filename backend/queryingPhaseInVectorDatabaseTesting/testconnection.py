import time
from pymilvus import MilvusClient
from globalConstants import COLLECTIONNAME
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("MILVUS_URI")
USER = os.getenv("MILVUS_USER")
PASSWORD = os.getenv("MILVUS_PASSWORD")

def test(number):
    client = MilvusClient(uri = URI, token = os.getenv("TOKEN"))
    print(client)
    try : 
        if number == 1:
            print("\n--- Phase 3: Build index (post-ingestion, sync) ---")

            listofcollections = client.list_collections()
            print(f"Collections in Milvus: {listofcollections}")

            loadstate = client.get_load_state(collection_name = COLLECTIONNAME)

            idx_params = client.prepare_index_params()
            idx_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 64})
            t0 = time.time()
            client.create_index(COLLECTIONNAME, idx_params)
            t1 = time.time()
            print(f"Index build (sync) took {t1 - t0:.2f}s")
            client.close()
        else:
            dbs = client.list_databases()
            print(f"Successfully listed databases: {dbs}")

            user_info = client.describe_user(user_name="db_46f4508aa296b4c")
            print(f"User Details: {user_info}")
            current_user = client.list_users() 
            print(f"Authenticated as: {current_user}")
            client.grant_role(user_name="db_46f4508aa296b4c", role_name="admin")

            roles = client.list_roles()
            print(f"Roles assigned to user: {roles}")
            print("Altering the properties")
            client.alter_database_properties(
            db_name = COLLECTIONNAME,
            properties = {
                "database.force.deny.reading": False
                }
            )
            print("done")
    finally:
        try :
            client.close()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    test(7)