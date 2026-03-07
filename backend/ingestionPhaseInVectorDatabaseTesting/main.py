# import os
# import queue
# import threading
# import time
# import numpy as np
# from dotenv import load_dotenv

# # Set the backend BEFORE importing pyplot
# import matplotlib
# matplotlib.use('TkAgg') 
# import matplotlib.pyplot as plt

# from globalConstants import NUMVECTORS, DIMENSIONS, NUMWORKERS, BATCHSIZE, INTERVAL, INCREASETHEWORKERBY
# from pymilvus import MilvusClient

# # Load environment variables
# load_dotenv()

# uri = os.getenv("MILVUS_URI")
# user = os.getenv("MILVUS_USERNAME")
# password = os.getenv("MILVUSPASSWORD")

# # --- GLOBAL STATE ---
# ingestion_queue = queue.Queue(maxsize=100)
# all_worker_threads = []
# stop_scaling = threading.Event()
# ingestion_finished = threading.Event()

# TOTAL_INSERTED = 0
# metrics_lock = threading.Lock()
# plot_data = {'time': [], 'throughput': [], 'worker_count': []}

# # --- WORKER LOGIC ---
# def milvus_worker(worker_id):
#     print(f"[{worker_id}] Initializing client...")
#     try:
#         worker_client = MilvusClient(uri=uri, user=user, password=password, timeout=100000)
#         while True:
#             batch = ingestion_queue.get()
#             if batch is None:
#                 ingestion_queue.task_done()
#                 break
#             batch_vectors, batch_ids = batch
            
#             try:
#                 data_to_insert = [{"vector": vec.tolist(), "id": id_val} for vec, id_val in zip(batch_vectors, batch_ids)]
#                 worker_client.insert(
#                     collection_name="stressTestingDuringIngestion_copy",
#                     data=data_to_insert
#                 )
#                 with metrics_lock:
#                     global TOTAL_INSERTED
#                     TOTAL_INSERTED += len(batch)
#             except Exception as e:
#                 print(f"[{worker_id}] Insert Error: {e}")
#             finally:
#                 ingestion_queue.task_done()
#         worker_client.close()
#     except Exception as e:
#         print(f"[{worker_id}] Connection failed: {e}")

# # --- SCALING LOGIC ---
# def scaler_thread(increment_by, interval_seconds):
#     while not stop_scaling.is_set():
#         time.sleep(interval_seconds)
#         if stop_scaling.is_set():
#             break
            
#         current_total = len(all_worker_threads)
#         print(f"\n>>> [SCALER] Adding {increment_by} workers. Total active: {current_total + increment_by} <<<\n")
        
#         for i in range(increment_by):
#             new_id = f"W-{current_total + i}"
#             t = threading.Thread(target=milvus_worker, args=(new_id,), daemon=True)
#             t.start()
#             all_worker_threads.append(t)

# # --- PLOTTING LOGIC (MUST RUN IN MAIN THREAD) ---
# def start_realtime_plot():
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(12, 6))
#     line, = ax.plot([], [], 'b-', lw=2, label='Instant Throughput (Vectors/s)')
#     ax.set_xlabel('Elapsed Time (s)')
#     ax.set_ylabel('Vectors Per Second')
#     ax.set_title('Real-Time Ingestion Stress Test (Sliding Window)')
#     ax.grid(True, alpha=0.3)
#     ax.legend()
    
#     start_time = time.time()
#     last_total = 0
#     last_time = start_time
#     last_worker_count = NUMWORKERS

#     # Loop until ingestion is finished
#     while not ingestion_finished.is_set():
#         current_time = time.time()
#         delta_time = current_time - last_time
        
#         if delta_time >= 1.0: # Calculate every 1 second
#             with metrics_lock:
#                 current_total = TOTAL_INSERTED
#                 current_workers = len(all_worker_threads)
            
#             # Sliding window throughput
#             instant_throughput = (current_total - last_total) / delta_time
#             elapsed = current_time - start_time
            
#             plot_data['time'].append(elapsed)
#             plot_data['throughput'].append(instant_throughput)
            
#             # Update plot
#             line.set_data(plot_data['time'], plot_data['throughput'])
#             ax.relim()
#             ax.autoscale_view()
            
#             # Annotation for new workers
#             if current_workers > last_worker_count:
#                 ax.annotate(f'{current_workers} Workers', 
#                             xy=(elapsed, instant_throughput),
#                             xytext=(15, 15), textcoords='offset points',
#                             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
#                 last_worker_count = current_workers

#             last_total = current_total
#             last_time = current_time

#         plt.pause(0.1) 
    
#     plt.ioff()
#     print("Ingestion finished. Showing final plot...")
#     plt.show()

# # --- INGESTION ORCHESTRATOR ---
# def run_ingestion_pipeline(normalized_vectors):
#     num_total = len(normalized_vectors)
#     global all_worker_threads
#     all_worker_threads = []
#     stop_scaling.clear()
#     ingestion_finished.clear()

#     print(f"\n--- Starting Ingestion Pipeline ---")
#     print(f"Total Vectors: {num_total} | Initial Workers: {NUMWORKERS} | Batch Size: {BATCHSIZE}")

#     # 1. Start Initial Workers
#     for i in range(NUMWORKERS):
#         t = threading.Thread(target=milvus_worker, args=(f"W-{i}",), daemon=True)
#         t.start()
#         all_worker_threads.append(t)

#     # 2. Start Scaler
#     ramp_logic = threading.Thread(target=scaler_thread, args=(INCREASETHEWORKERBY, INTERVAL), daemon=True)
#     ramp_logic.start()

#     # 3. Producer Loop
#     start_time = time.time()
#     for i in range(0, num_total, BATCHSIZE):
#         batch_slice = normalized_vectors[i : i + BATCHSIZE]
#         batch_ids = list(range(i, i + len(batch_slice)))
#         ingestion_queue.put((batch_slice, batch_ids))
    
#     # 4. Cleanup
#     ingestion_queue.join() 
#     stop_scaling.set()
    
#     for _ in range(len(all_worker_threads)):
#         ingestion_queue.put(None)
    
#     for t in all_worker_threads:
#         t.join()

#     print(f"Ingestion logic complete. Total Time: {time.time() - start_time:.2f}s")
#     ingestion_finished.set()

# # --- MAIN ENTRY POINT ---
# def dotesting():
#     print("--- Phase 1: Data Generation ---")
#     vectors = np.random.randn(NUMVECTORS, DIMENSIONS).astype(np.float32)
#     normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
#     print(f"Memory Usage: {normalized_vectors.nbytes / (1024**3):.2f} GB")

#     print("\n--- Phase 2: Milvus Stress Test ---")
    
#     # Start ingestion in background
#     t_ingest = threading.Thread(target=run_ingestion_pipeline, args=(normalized_vectors,), daemon=True)
#     t_ingest.start()

#     # Run Plotter in MAIN THREAD
#     try:
#         start_realtime_plot()
#     except KeyboardInterrupt:
#         print("\nInterrupted by user.")
#     finally:
#         del vectors
#         del normalized_vectors

# if __name__ == "__main__":
#     dotesting()
import os
import queue
import threading
import time
import numpy as np
from dotenv import load_dotenv

# Set the backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# Constants (Assumed from your globalConstants.py)
from globalConstants import NUMVECTORS, DIMENSIONS, NUMWORKERS, BATCHSIZE, INTERVAL, INCREASETHEWORKERBY, COLLECTIONNAME
from pymilvus import MilvusClient, DataType

load_dotenv()

uri = os.getenv("MILVUS_URI")
user = os.getenv("MILVUS_USERNAME")
password = os.getenv("MILVUSPASSWORD")

COLLECTION_NAME = COLLECTIONNAME

# --- GLOBAL STATE ---
ingestion_queue = queue.Queue(maxsize=100)
all_worker_threads = []
stop_scaling = threading.Event()
ingestion_finished = threading.Event()

TOTAL_INSERTED = 0
metrics_lock = threading.Lock()
plot_data = {'time': [], 'throughput': [], 'total' : [], 'worker_count': []}

# --- PHASE 0: ARCHITECTURE CLEANUP ---
def prepare_milvus_clean_slate():
    """Ensures no zombie connections, no active indexing, and a fresh collection."""
    print("\n--- Phase 0: Resetting Milvus Architecture ---")
    client = MilvusClient(uri=uri, user=user, password=password)
    
    collectionsAvailable = client.list_collections()
    print("Available collections:", collectionsAvailable)

    if COLLECTION_NAME in collectionsAvailable:
        if client.has_collection(COLLECTION_NAME):
            print("Collection exists. Checking for active tasks and clearing state...")
            print("Checking for active background tasks...")
            # Check for indexing lag from previous runs
            indices = client.list_indexes(COLLECTION_NAME)
            for idx in indices:
                try:
                    # Wait for previous index to finish so we don't drop while CPU is at 100%
                    print(f"Waiting for previous index '{idx}' to complete before dropping collection")
                    client.wait_for_index_build_complete(COLLECTION_NAME, idx, timeout=60)
                except:
                    print(f"Index '{idx}' did not complete in time. Proceeding with drop anyway, but be aware of potential CPU spikes")
            
            print("Dropping collection to clear Proxy/DataNode buffers...")
            client.drop_collection(collection_name = COLLECTION_NAME)
            # Crucial: Let the distributed system sync the deletion
            print("Waiting for architecture to stabilize after drop, hence waiting for five seconds")
            time.sleep(5)

    print(f"Creating fresh collection: {COLLECTION_NAME} (No Indexing Yet)")
    schema = client.create_schema()
    schema.add_field(field_name="primaryKey", is_primary=True, auto_id=True, datatype=DataType.INT64)
    schema.add_field(field_name="id", datatype=DataType.INT64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSIONS)
    
    # We create the collection but DO NOT create an index here to maximize raw V/S
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    client.close()
    print("Server is idle and ready.\n")

# --- WORKER LOGIC ---
def milvus_worker(worker_id):
    worker_client = None
    try:
        # Short timeout to prevent zombie hangs
        worker_client = MilvusClient(uri=uri, user=user, password=password, timeout=10)
        while True:
            batch = ingestion_queue.get()
            if batch is None: # Poison pill
                break
            
            try:
                batch_vectors, batch_ids = batch
                data_to_insert = [{"vector": vec.tolist(), "id": id_val} for vec, id_val in zip(batch_vectors, batch_ids)]
                
                worker_client.insert(
                    collection_name=COLLECTION_NAME,
                    data=data_to_insert
                )
                
                with metrics_lock:
                    global TOTAL_INSERTED
                    TOTAL_INSERTED += len(batch_vectors)
            except Exception as e:
                print(f"[{worker_id}] Insert Error: {e}")
            finally:
                ingestion_queue.task_done()
    except Exception as e:
        print(f"[{worker_id}] Connection/Critical Error: {e}")
    finally:
        if worker_client:
            worker_client.close()
            print(f"[{worker_id}] Connection closed.")
        # Ensure queue doesn't hang even if client failed to init
        if ingestion_queue.unfinished_tasks > 0:
            try: ingestion_queue.task_done()
            except: pass

# --- SCALING & PLOTTING ---
# (Logic remains same as your working version but includes current_workers check)
def scaler_thread(increment_by, interval_seconds):
    while not stop_scaling.is_set():
        time.sleep(interval_seconds)
        if stop_scaling.is_set(): break
        current_total = len(all_worker_threads)
        for i in range(increment_by):
            new_id = f"W-{current_total + i}"
            t = threading.Thread(target=milvus_worker, args=(new_id,), daemon=True)
            print(f"\n>>> [SCALER] Adding worker {new_id}. Total active: {current_total + 1} <<<\n")
            t.start()
            all_worker_threads.append(t)
        print(f"\n>>> [FINAL WORKER WHICH IS CURRENTLY WORKING] Total active workers: {len(all_worker_threads)} <<<\n")

def start_realtime_plot():
    plt.ion()
    # fig, ax = plt.subplots(figsize=(12, 6))
    # line, = ax.plot([], [], 'g-', lw=2, label='Raw Ingestion (No Indexing)')
    # ax.set_xlabel('Seconds')
    # ax.set_ylabel('Vectors/Sec')
    # ax.set_title(f'Stress Test: {DIMENSIONS} Dims | Raw Throughput')
    # ax.grid(True, alpha=0.2)

    # Replace the old fig, ax lines with this:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Throughput Line (Top Graph)
    line1, = ax1.plot([], [], 'g-', lw=2, label='Vectors/Sec')
    ax1.set_ylabel('Vectors Per Second')
    ax1.set_title('Real-Time Ingestion Performance')
    ax1.grid(True, alpha=0.2)

    # Total Count Line (Bottom Graph)
    line2, = ax2.plot([], [], 'b-', lw=2, label='Total Inserted')
    ax2.set_xlabel('Seconds')
    ax2.set_ylabel('Total Vectors')
    ax2.grid(True, alpha=0.2)
    
    start_time = time.time()
    last_total, last_time = 0, start_time
    last_worker_count = NUMWORKERS

    while not ingestion_finished.is_set():
        now = time.time()
        delta = now - last_time
        if delta >= 1.0:
            with metrics_lock:
                curr_total, curr_workers = TOTAL_INSERTED, len(all_worker_threads)
            
            tput = (curr_total - last_total) / delta
            plot_data['time'].append(now - start_time)
            plot_data['throughput'].append(tput)
            plot_data['total'].append(curr_total)
            
            line1.set_data(plot_data['time'], plot_data['throughput'])
            line2.set_data(plot_data['time'], plot_data['total'])

            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            
            if curr_workers > last_worker_count:
                ax1.annotate(f'{curr_workers}W', xy=(now-start_time, tput), color='red')
                last_worker_count = curr_workers
            
            last_total, last_time = curr_total, now
        plt.pause(0.1)
    plt.ioff(); plt.show()

# --- ORCHESTRATOR ---
def run_ingestion_pipeline(normalized_vectors):
    global all_worker_threads
    all_worker_threads = []
    stop_scaling.clear()
    ingestion_finished.clear()

    # Start Workers
    for i in range(NUMWORKERS):
        t = threading.Thread(target=milvus_worker, args=(f"W-{i}",), daemon=True)
        t.start()
        all_worker_threads.append(t)
    print(f"\n>>> [FINAL WORKER WHICH IS CURRENTLY WORKING] Total active workers: {len(all_worker_threads)} <<<\n")

    threading.Thread(target=scaler_thread, args=(INCREASETHEWORKERBY, INTERVAL), daemon=True).start()

    # Feed Queue
    for i in range(0, len(normalized_vectors), BATCHSIZE):
        batch = (normalized_vectors[i : i + BATCHSIZE], list(range(i, i + len(normalized_vectors[i : i + BATCHSIZE]))))
        ingestion_queue.put(batch)
    
    ingestion_queue.join() 
    stop_scaling.set()
    for _ in range(len(all_worker_threads)): ingestion_queue.put(None)
    for t in all_worker_threads: t.join()
    ingestion_finished.set()

def dotesting():
    print("--- Phase 1: Data Generation ---")
    vectors = np.random.randn(NUMVECTORS, DIMENSIONS).astype(np.float32)
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    prepare_milvus_clean_slate()

    print("--- Phase 2: High-Pressure Ingestion ---")
    t_ingest = threading.Thread(target=run_ingestion_pipeline, args=(normalized_vectors,), daemon=True)
    t_ingest.start()

    try:
        start_realtime_plot()
        
        # PHASE 3: Optional - Create Index after ingestion is done
        print("\n--- Phase 3: Building Index (Post-Ingestion) ---")
        client = MilvusClient(uri=uri, user=user, password=password)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 64})
        start_time = time.time()
        client.create_index(COLLECTION_NAME, index_params)
        end_time = time.time()
        print(f"Index built in {end_time - start_time:.2f} seconds")
        print("Indexing started after ingestion, so raw V/S was unaffected. This phase is just to measure indexing time under load")
        client.close()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        del vectors, normalized_vectors

if __name__ == "__main__":
    dotesting()