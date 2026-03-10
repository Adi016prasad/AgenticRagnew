import os
import time
import numpy as np
from locust import User, task, between, events
from pymilvus import MilvusClient
from dotenv import load_dotenv

# Ensure COLLECTIONNAME is imported correctly
try:
    from globalConstants import COLLECTIONNAME
except ImportError:
    COLLECTIONNAME = "your_default_collection"

load_dotenv()

# Pre-generate vectors once for memory efficiency
SHARED_VECTORS = np.random.standard_normal(size=(1000, 4096)).astype(np.float32)

class MilvusUser(User):
    # Set wait_time so tasks aren't executed in an infinite loop without delay
    wait_time = between(1, 3) 

    def on_start(self):
        """Initializes connection when the user begins the test in the UI."""
        self.client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("TOKEN")
        )

    def on_stop(self):
        """Closes connection when the test is stopped in the UI."""
        if hasattr(self, 'client'):
            self.client.close()

    @task
    def search_vector(self):
        idx = np.random.randint(0, len(SHARED_VECTORS))
        query_vec = SHARED_VECTORS[idx].tolist()
        
        start_time = time.perf_counter()
        
        try:
            result = self.client.search(
                collection_name=COLLECTIONNAME,
                anns_field="vector",
                data=[query_vec],
                limit=5,
                search_params={"metric_type": "COSINE"}
            )
            
            total_time = (time.perf_counter() - start_time) * 1000
            events.request.fire(
                request_type="Milvus",
                name="search",
                response_time=total_time,
                response_length=len(str(result)),
                exception=None
            )
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            events.request.fire(
                request_type="Milvus",
                name="search",
                response_time=total_time,
                response_length=0,
                exception=e
            )