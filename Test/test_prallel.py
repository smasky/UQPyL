import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tqdm
import time
def worker(num):
    temp=0
    for i in range(num):
        temp+=i
    return temp
if __name__=='__main__':
    a=time.time()
    # for _ in range(100):
    #     print(worker(10000))
    with ThreadPoolExecutor(max_workers=6) as exe:
        futures=[]
        for _ in range(10000):
            future=exe.submit(worker, 10000)
            futures.append(future)
        
        for future in as_completed(futures):
            print(future.result())
    b=time.time()
    print(b-a)
