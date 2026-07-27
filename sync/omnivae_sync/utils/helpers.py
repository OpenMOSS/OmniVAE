import os

def waiting_for_debug(ip, port):
    import debugpy
    rank = os.environ.get("RANK", "0")
    debugpy.listen((ip, port)) # 把这边的 localhost 改成集群节点 ip
    print(f"[rank = {rank}] Waiting for debugger attach...")
    debugpy.wait_for_client()
    print(f"[rank = {rank}] Debugger attached")