import os
import re
import numa
import logging


logging.basicConfig(level=logging.INFO)
ilogger = logging.getLogger(__name__)

# 设置日志级别
ilogger.setLevel(logging.INFO)


def detect_hg_dcu_get_numa_affinity():
    """
    执行 hy-smi --showtopo 命令，解析输出，返回 dcu_id 到 numa_affinity 的映射字典。
    
    返回:
        dict: {dcu_id: numa_affinity, ...}，例如 {0: 3, 1: 1, 2: 1, ...}
               如果解析失败或命令出错，返回空字典。
    """
    result = os.popen("hy-smi --showtopo").read().strip()
    
    # 用于存储 dcu_id -> numa_affinity 的映射
    dcu_to_numa = {}
    
    # 按行分割输出
    lines = result.splitlines()
    
    for line in lines:
        line = line.strip()
        
        # 匹配形如 "HCU[0]          : Numa Affinity:  3" 的行
        # 注意：这里使用更灵活的空白符匹配
        match = re.match(r'HCU\[(\d+)\]\s*:\s*Numa Affinity:\s*(\d+)', line)
        if match:
            dcu_id = int(match.group(1))
            numa_affinity = int(match.group(2))
            dcu_to_numa[dcu_id] = numa_affinity
    
    return dcu_to_numa




# 设置当前进程绑定到 NUMA 节点
def auto_bind_to_numa():
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    dcu_to_numa = detect_hg_dcu_get_numa_affinity()
    if local_rank not in dcu_to_numa:
        ilogger.warning("local_rank is not in map, it will not bind to numa!")
        return 
    
    numa_node = dcu_to_numa[local_rank]

    # 未配置环境变量或配置错误则不做绑定，TODO：根据topo自动绑定方案
    if numa_node < 0:
        ilogger.warning("numa node is unvalid, it will not bind to numa!")
        return

    if numa_node > numa.get_max_node():
        raise ValueError(f"NUMA node {numa_node} is not available.")
    
    # 使用首选mem node;
    numa.set_preferred(numa_node)
    cpus = list(numa.node_to_cpus(numa_node))

    #numa.bind([numa_node]) # mem和cpu核心强制绑定，mem不支持fallback到其他节点

    pid = os.getpid()
    os.sched_setaffinity(pid, cpus)
    ilogger.info("########## %d process(rank%s) is running on CPU(s): %s", pid, str(int(os.getenv("LOCAL_RANK", 0))), str(os.sched_getaffinity(pid)))
    ilogger.info("########## %d process(rank%s) is running on memnode(s): %s", pid, str(int(os.getenv("LOCAL_RANK", 0))), str(numa.get_membind()))


# =============== 使用示例 ===============
if __name__ == "__main__":
    # 绑定当前进程到指定 NUMA 节点
    auto_bind_to_numa()

    numa_map = detect_hg_dcu_get_numa_affinity()
    print("DCU ID 到 NUMA Affinity 映射:")
    for dcu_id, numa in sorted(numa_map.items()):
        print(f"DCU[{dcu_id}] -> NUMA {numa}")
