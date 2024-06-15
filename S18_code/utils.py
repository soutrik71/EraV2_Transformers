import torch
import gc
import time

start_time = None


def start_timer():
    global start_time
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Max memory used in start: {max_memory:.2f}GB")


def end_timer(local_msg):
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"{local_msg} -> Max memory used at end: {max_memory:.2f}GB, Time Taken: {time.time() - start_time:.2f} sec"
    )
