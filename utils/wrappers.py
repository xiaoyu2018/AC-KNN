
# 计时器
def time_counter(func):
    from time import time
    def wrapper(*args,**kwargs):
        start=time()
        res=func(*args,**kwargs)
        end=time()
        print(f"------------------{func.__name__}-----------------")
        print(f"time consumed: {end-start}s")
        print("------------------------------------------")
        return res
    return wrapper