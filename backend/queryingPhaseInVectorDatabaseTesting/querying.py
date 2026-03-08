import json
import os
import queue
import csv
import time
import numpy as np
from pymilvus import AsyncMilvusClient, MilvusClient
from globalConstants import COLLECTIONNAME, DIMENSIONS, NUMVECTORS, NUMWORKERS, TOPK
from dotenv import load_dotenv
import asyncio

load_dotenv()

URI = os.getenv("MILVUS_URI")
USER = os.getenv("MILVUS_USER")
PASSWORD = os.getenv("MILVUS_PASSWORD")
TOKEN = os.getenv("TOKEN")

def prepare_milvus_clean_slate(collection_name : str, dimension : int):
    client = MilvusClient(uri = URI, token = TOKEN)

    try :
        collectionList = client.list_collections()
        if collection_name in collectionList :
            print(f"Collection {collection_name} already exists. Deleting it for a clean slate")

            nameOfIndexes = client.list_indexes(collection_name = collection_name)

            for index in nameOfIndexes:
                try :
                    print(f"Deleting index {index['name']} for collection {collection_name}")
                    client.drop_index(collection_name = collection_name, index_name = index['name'])
                except Exception as e:
                    print(f"Error deleting index {index['name']} for collection {collection_name}: {e}")
            
            try :
                print(f"Deleting collection {collection_name}")
                client.delete(collection_name = collection_name, filter = "id >= 0")
            except Exception as e:
                print(f"Error deleting collection {collection_name}: {e}")
            client.drop_collection(collection_name)
    except Exception as e:
        print(f"Error preparing collection {collection_name}: {e}")
    finally :
        try :
            client.close()
        except Exception as e:
            print(f"Error closing the client: {e}")

class AsyncQuery:
    def __init__(self, collection_name : str, vectors : np.ndarray, top_k : int):
        self.collection_name = collection_name
        self.vectors = vectors
        self.top_k = top_k
        self.latencies = []
        self.num_workers = NUMWORKERS
        self.lock = asyncio.Lock()
        self.start_time = None
        self.last_report_total = 0
        self.last_report_time = None
        self.queue = asyncio.Queue()
    
    async def query(self, worker_id : int):
        client = AsyncMilvusClient(uri = URI, token = TOKEN)
        try :
            while True :
                item = await self.queue.get()

                if item is None :
                    self.queue.task_done()
                    break

                try :
                    start = time.perf_counter()
                    await client.search(
                        collection_name = self.collection_name,
                        anns_field="vector",
                        data=[item],
                        limit=self.top_k,
                        search_params={"metric_type": "COSINE"}
                    )
                    end = time.perf_counter()
                    totalTimeTaken = end - start

                    async with self.lock:
                        self.latencies.append(totalTimeTaken)
                except Exception as e:
                    print(f"Worker {worker_id} encountered an error: {e}")
                finally:
                    self.queue.task_done()
        finally :
            try :
                await client.close()
            except Exception as e:
                print(f"Worker {worker_id} encountered an error while closing the client: {e}")

    async def producer(self):
        """Feeds individual vectors into the queue."""
        for vec in self.vectors:
            await self.queue.put(vec.tolist())
        
        for _ in range(self.num_workers):
            await self.queue.put(None)
    
    async def reporter(self, interval: float = 1.0):
        """Periodically prints real-time instantaneous QPS to the console."""
        self.last_report_time = time.perf_counter()
        self.last_report_total = 0
        total_vectors_to_query = len(self.vectors)
        
        while True:
            await asyncio.sleep(interval)
            
            async with self.lock:
                current_total = len(self.latencies)
                
            now = time.perf_counter()
            delta_q = current_total - self.last_report_total
            delta_t = now - self.last_report_time
            
            instant_qps = delta_q / delta_t if delta_t > 0 else 0.0
            elapsed = now - self.start_time if self.start_time else 0.0
            
            print(f"[{elapsed:.1f}s] Queries done: {current_total}/{total_vectors_to_query} | Instant QPS: {instant_qps:.1f} q/s")
            
            self.last_report_time = now
            self.last_report_total = current_total
            
            if current_total >= total_vectors_to_query:
                break

    async def run(self):
        """Orchestrate workers + producer + reporter and return elapsed seconds."""
        workers = [asyncio.create_task(self.query(i)) for i in range(self.num_workers)]
        self.start_time = time.perf_counter()
        reporter_task = asyncio.create_task(self.reporter())
        await self.producer()
        await asyncio.gather(*workers)
        await reporter_task
        elapsed = time.perf_counter() - self.start_time
        return elapsed

def summarize_and_write(latencies, elapsed, out_csv):
    """Calculates percentiles and QPS, prints a summary, and logs it to a CSV."""
    if not latencies:
        print("No latencies recorded. Skipping summary.")
        return

    # Convert to numpy array for calculating percentiles
    lat_arr = np.array(latencies)
    
    # Statistical calculations
    avg_latency = np.mean(lat_arr)
    p25 = np.percentile(lat_arr, 25)
    p50 = np.percentile(lat_arr, 50)
    p95 = np.percentile(lat_arr, 95)
    p99 = np.percentile(lat_arr, 99)
    
    total_queries = len(latencies)
    overall_qps = total_queries / elapsed if elapsed > 0 else 0.0
    
    # Print real-time summary to terminal
    print("\n" + "="*40)
    print("--- Final Querying Summary ---")
    print(f"Total Queries Executed : {total_queries}")
    print(f"Total Time Elapsed     : {elapsed:.2f} seconds")
    print(f"Overall QPS            : {overall_qps:.2f} queries/sec")
    print(f"Average Latency        : {avg_latency:.6f} seconds")
    print(f"p25 Latency            : {p25:.6f} seconds")
    print(f"p50 Latency            : {p50:.6f} seconds")
    print(f"p95 Latency            : {p95:.6f} seconds")
    print(f"p99 Latency            : {p99:.6f} seconds")
    print("="*40 + "\n")
    
    # Write/Append to CSV
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, mode='a', newline='') as csvfile:
        fieldnames = ['Total_Queries', 'Elapsed_Time', 'Overall_QPS', 'Avg_Latency', 'p25', 'p50', 'p95', 'p99']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'Total_Queries': total_queries,
            'Elapsed_Time': round(elapsed, 4),
            'Overall_QPS': round(overall_qps, 2),
            'Avg_Latency': round(avg_latency, 6),
            'p25': round(p25, 6),
            'p50': round(p50, 6),
            'p95': round(p95, 6),
            'p99': round(p99, 6)
        })
    print(f"Successfully saved summary to {out_csv}")

def generate_normalized_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def main():
    print("--- Phase 1: Data Generation ---")
    vectors = generate_normalized_vectors(NUMVECTORS, DIMENSIONS)
    print(f"Generated {len(vectors)} vectors ({vectors.nbytes / (1024**3):.2f} GB)")

    print("\n--- Phase 2: Async Querying (no dynamic scaling) ---")
    runner = AsyncQuery(collection_name = COLLECTIONNAME, vectors = vectors, top_k = TOPK)
    elapsed = asyncio.run(runner.run())

    summarize_and_write(runner.latencies, elapsed, out_csv=f"latencies{NUMWORKERS}{TOPK}.csv")


if __name__ == "__main__":
    main()