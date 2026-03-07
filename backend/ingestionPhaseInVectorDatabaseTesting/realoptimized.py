from http import client
import os
import time
import asyncio
import csv
from collections import namedtuple
from statistics import mean
from dotenv import load_dotenv
import random

import numpy as np

# sync client used only for collection prepare/cleanup (runs before event loop)
from pymilvus import MilvusClient, DataType
from pymilvus import AsyncMilvusClient

# import your constants
from globalConstants import NUMVECTORS, DIMENSIONS, NUMWORKERS, BATCHSIZE, COLLECTIONNAME

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USERNAME")
MILVUS_PASS = os.getenv("MILVUSPASSWORD")

LatencyRecord = namedtuple("LatencyRecord", ["batch_index", "num_vectors", "latency_s", "workers_at_time"])


# -------------------------- helper: prepare collection (sync) --------------------------
def prepare_milvus_clean_slate(collection_name: str, dim: int):
    """Ensures no zombie connections, no active indexing, and a fresh collection."""
    print("\n--- Phase 0: Resetting Milvus Architecture ---")
    client = MilvusClient(uri= MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASS)
    
    collectionsAvailable = client.list_collections()
    print("Available collections:", collectionsAvailable)

    if collection_name in collectionsAvailable:
        if client.has_collection(collection_name):
            print("Collection exists. Checking for active tasks and clearing state...")
            print("Checking for active background tasks...")
            # Check for indexing lag from previous runs

            indices = client.list_indexes(collection_name)
            print(f"Number of indexes found are {len(indices)}")
            print("Indices found are:", indices)
            for idx in indices:
                try:
                    print(f"Waiting for previous index '{idx}' to complete before dropping collection")
                    client.drop_index(collection_name = COLLECTIONNAME, index_name = idx, timeout = 10000)
                    print("Index is dropped")
                except Exception as e :
                    print(f"Index '{idx}' did not complete in time. Proceeding with drop anyway, but be aware of potential CPU spikes {e}")
            
            try :
                client.delete(collection_name=COLLECTIONNAME, filter = "id >= 0")
            except Exception as e:
                print(f"Exception occurred while deleting the data {e}")
            print("Dropping collection to clear Proxy/DataNode buffers...")
            client.drop_collection(collection_name = collection_name)
            # Crucial: Let the distributed system sync the deletion
            print("Waiting for architecture to stabilize after drop, hence waiting for five seconds")
            time.sleep(5)

    print(f"Creating fresh collection: {collection_name} (No Indexing Yet)")
    schema = client.create_schema()
    schema.add_field(field_name="primaryKey", is_primary=True, auto_id=True, datatype=DataType.INT64)
    schema.add_field(field_name="id", datatype=DataType.INT64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSIONS)
    
    # We create the collection but DO NOT create an index here to maximize raw V/S
    client.create_collection(collection_name=collection_name, schema=schema)
    client.close()
    print("Server is idle and ready.\n")


# -------------------------- small PgBouncer-style pool & retry --------------------------
class MilvusClientPool:
    """
    Simple asyncio-based pool of AsyncMilvusClient instances.
    Acts like PgBouncer but for gRPC clients to :contentReference[oaicite:1]{index=1}.
    """
    def __init__(self, uri, user, password, pool_size=5, timeout=60000):
        self.uri = uri
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.timeout = timeout
        self._queue = asyncio.Queue()
        self._initialized = False

    async def init(self):
        if self._initialized:
            return
        for _ in range(self.pool_size):
            client = AsyncMilvusClient(uri=self.uri, user=self.user, password=self.password, timeout=self.timeout)
            await self._queue.put(client)
        self._initialized = True

    async def acquire(self):
        return await self._queue.get()

    async def release(self, client):
        await self._queue.put(client)

    async def close_all(self):
        while not self._queue.empty():
            client = await self._queue.get()
            try:
                await client.close()
            except Exception:
                pass


async def insert_with_retry(client, collection_name, payload, max_retries=5, base_backoff=0.5):
    """
    Insert with jittered exponential backoff retry. Returns latency seconds on success.
    Raises last exception on final failure.
    """
    backoff = base_backoff
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            await client.insert(collection_name=collection_name, data=payload)
            return time.perf_counter() - t0
        except Exception as e:
            msg = str(e).lower()
            transient = ("unavailable" in msg) or ("reset" in msg) or ("connect error" in msg) or ("rpcerror" in msg)
            if attempt == max_retries or not transient:
                raise
            # jittered exponential backoff
            sleep_for = backoff * (0.5 + random.random())
            await asyncio.sleep(sleep_for)
            backoff *= 2


# -------------------------- async ingestion benchmark (updated to use pool & retries) --------------------------
class AsyncIngest:
    def __init__(self, collection_name: str, vectors: np.ndarray, batch_size: int, num_workers: int, pool_size: int = None):
        self.collection = collection_name
        self.vectors = vectors
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = asyncio.Queue(maxsize=512)
        self.latencies = []
        self.total_inserted = 0
        self._lock = asyncio.Lock()
        self.total_batches = (len(vectors) + batch_size - 1) // batch_size
        self._start_time = None
        self._last_report_total = 0
        self._last_report_time = None

        # pool will be initialized in run()
        self.pool = None
        self._pool_size_override = pool_size

    async def _worker(self, worker_id: int):
        """
        Worker borrows a client from the pool, uses insert_with_retry, then returns it.
        """
        try:
            while True:
                item = await self.queue.get()
                if item is None:  # poison pill
                    self.queue.task_done()
                    break

                batch_index, batch_vectors, batch_ids, workers_at_enqueue = item
                payload = [{"vector": v.tolist(), "id": int(i)} for v, i in zip(batch_vectors, batch_ids)]

                client = None
                try:
                    client = await self.pool.acquire()
                    latency = await insert_with_retry(client, self.collection, payload, max_retries=5, base_backoff=0.25)
                    async with self._lock:
                        self.latencies.append(LatencyRecord(batch_index, len(batch_vectors), latency, workers_at_enqueue))
                        self.total_inserted += len(batch_vectors)
                except Exception as e:
                    print(f"[worker-{worker_id}] insert error: {e}")
                finally:
                    if client is not None:
                        await self.pool.release(client)
                    self.queue.task_done()
        except asyncio.CancelledError:
            # graceful exit if cancelled
            pass

    async def _producer(self):
        """Push batches to the queue. Records workers_at_enqueue as the current worker count."""
        idx = 0
        batch_idx = 0
        total = len(self.vectors)
        counter = 0
        while idx < total:
            end = min(idx + self.batch_size, total)
            batch_vectors = self.vectors[idx:end]
            batch_ids = list(range(idx, end))
            # record current concurrent workers at time of enqueue (approx)
            await self.queue.put((batch_idx, batch_vectors, batch_ids, self.num_workers))
            batch_idx += 1
            idx = end

            # gentle producer pacing to avoid instant queue flooding:
            counter += 1
            if counter % 50 == 0:
                await asyncio.sleep(0.005)

        # producer done; wait for queue to be processed
        await self.queue.join()
        # send poison pills to workers so they can exit
        for _ in range(self.num_workers):
            await self.queue.put(None)

    async def _reporter(self, interval: float = 1.0):
        """Periodically print simple throughput stats to console (no GUI)."""
        self._last_report_time = time.perf_counter()
        self._last_report_total = 0
        while True:
            await asyncio.sleep(interval)
            async with self._lock:
                current_total = self.total_inserted
            now = time.perf_counter()
            delta_v = current_total - self._last_report_total
            delta_t = now - self._last_report_time
            tput = delta_v / delta_t if delta_t > 0 else 0.0
            elapsed = now - self._start_time if self._start_time else 0.0
            print(f"[{elapsed:.1f}s] total_inserted={current_total}  instant_tput={tput:.1f} vec/s  batches_done={len(self.latencies)}/{self.total_batches}")
            self._last_report_time = now
            self._last_report_total = current_total
            # stop reporting when all batches done
            if len(self.latencies) >= self.total_batches:
                break

    async def run(self):
        """Orchestrate pool + workers + producer + reporter and return elapsed seconds."""
        # determine pool size: small relative to workers (tuneable)
        if self._pool_size_override is not None:
            pool_size = self._pool_size_override
        else:
            pool_size = min(8, max(1, self.num_workers // 4))

        # initialize pool
        self.pool = MilvusClientPool(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASS, pool_size=pool_size, timeout=60000)
        await self.pool.init()

        # spawn workers with a small stagger to avoid instant connection pressure on the pool
        workers = []
        for i in range(self.num_workers):
            workers.append(asyncio.create_task(self._worker(i)))
            if i % 10 == 0:
                # small stagger for every 10 workers
                await asyncio.sleep(0.01)

        # start reporter
        self._start_time = time.perf_counter()
        reporter_task = asyncio.create_task(self._reporter())

        # start producer (runs until it calls queue.join())
        await self._producer()

        # wait for workers to finish (they will exit after poison pills)
        await asyncio.gather(*workers, return_exceptions=True)

        # close pool clients
        await self.pool.close_all()

        # ensure reporter finishes
        await reporter_task
        elapsed = time.perf_counter() - self._start_time
        return elapsed


# -------------------------- utility: generate vectors & summary --------------------------
def generate_normalized_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def summarize_and_write(latencies, elapsed_s: float, out_csv: str):
    if not latencies:
        print("No latency records to summarize.")
        return
    lat_s = np.array([r.latency_s for r in latencies])
    counts = np.array([r.num_vectors for r in latencies])
    total_vectors = int(np.sum(counts))
    overall_tput = total_vectors / elapsed_s if elapsed_s > 0 else float("inf")

    p50 = float(np.percentile(lat_s, 50))
    p95 = float(np.percentile(lat_s, 95))
    p99 = float(np.percentile(lat_s, 99))

    print("\n----- SUMMARY -----")
    print(f"Total batches: {len(lat_s)}")
    print(f"Total vectors inserted: {total_vectors}")
    print(f"Elapsed (s): {elapsed_s:.3f}")
    print(f"Overall throughput (vectors/sec): {overall_tput:.2f}")
    print(f"Per-batch latency p50/p95/p99 (s): {p50:.6f} / {p95:.6f} / {p99:.6f}")
    print(f"Mean per-batch latency (s): {mean(lat_s):.6f}")

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_index", "num_vectors", "latency_s", "workers_at_time"])
        for r in latencies:
            w.writerow([r.batch_index, r.num_vectors, r.latency_s, r.workers_at_time])
    print(f"Wrote per-batch latencies to {out_csv}")


# -------------------------- main entry point --------------------------
def main():
    print("--- Phase 1: Data Generation ---")
    vectors = generate_normalized_vectors(NUMVECTORS, DIMENSIONS)
    print(f"Generated {len(vectors)} vectors ({vectors.nbytes / (1024**3):.2f} GB)")

    # Prepare collection synchronously (safe, runs before async ingestion)
    prepare_milvus_clean_slate(COLLECTIONNAME, DIMENSIONS)

    print("\n--- Phase 2: Async Ingestion (no dynamic scaling) ---")
    # optional: set a pool_size to override automatic selection: pool_size=6
    runner = AsyncIngest(collection_name=COLLECTIONNAME, vectors=vectors, batch_size=BATCHSIZE, num_workers=NUMWORKERS)
    elapsed = asyncio.run(runner.run())

    summarize_and_write(runner.latencies, elapsed, out_csv=f"latencies{NUMWORKERS}{NUMVECTORS}.csv")

    # Optional: build index after ingestion (sync)
    # try:
    #     print("\n--- Phase 3: Build index (post-ingestion, sync) ---")
    #     client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASS, timeout = 10000)
    #     idx_params = client.prepare_index_params()
    #     idx_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 64})
    #     t0 = time.time()
    #     client.create_index(COLLECTIONNAME, idx_params)
    #     t1 = time.time()
    #     print(f"Index build (sync) took {t1 - t0:.2f}s")
    #     client.close()
    # except Exception as e:
    #     print("Indexing failed or prepare_index API not available in this environment:", e)

    del vectors


if __name__ == "__main__":
    main()