import locust
import grpc.experimental.gevent as grpc_gevent
grpc_gevent.init_gevent()
import os
import time
import logging
import numpy as np
from locust import User, task, between, events
from pymilvus import MilvusClient
from dotenv import load_dotenv
from globalConstants import COLLECTIONNAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusTest")

load_dotenv()

SHARED_VECTORS = np.random.standard_normal(size=(1000, 4096)).astype(np.float32)

# 1. Define a global variable for the shared client
global_client = None

# 2. Initialize the client ONCE when the load test starts
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global global_client
    logger.info("Initializing global Milvus client (HTTP/2 Multiplexing)...")
    try:
        global_client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("TOKEN"),
            timeout=10
        )
        logger.info("Global Milvus client established successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize global client: {e}")
        environment.runner.quit()

# 3. Close the client safely when the test ends
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global global_client
    if global_client:
        global_client.close()
        logger.info("Global Milvus client closed.")

class MilvusUser(User):
    wait_time = between(0.1, 0.5)

    @task
    def search_vector(self):
        # Fail-safe in case the task triggers before the client is fully ready
        if not global_client:
            return 
            
        idx = np.random.randint(0, len(SHARED_VECTORS))
        query_vec = SHARED_VECTORS[idx].tolist()
        
        start_time = time.perf_counter()
        
        try:
            # All users share the same global_client
            result = global_client.search(
                collection_name=COLLECTIONNAME,
                anns_field="vector",
                data=[query_vec],
                limit=5,
                search_params={"metric_type": "COSINE"}
            )
            
            events.request.fire(
                request_type="Milvus",
                name="search",
                response_time=(time.perf_counter() - start_time) * 1000,
                response_length=len(str(result)),
                exception=None
            )
        except Exception as e:
            events.request.fire(
                request_type="Milvus",
                name="search",
                response_time=(time.perf_counter() - start_time) * 1000,
                response_length=0,
                exception=e
            )