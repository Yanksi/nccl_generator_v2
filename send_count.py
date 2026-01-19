import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="traces/Llama7B_trace/InterNode_MicroEvents_Dependency.goal", help="Path to the goal file to analyze")
args = parser.parse_args()
file = args.file
#

# file = "traces/Llama7B_trace/InterNode_MicroEvents_Dependency.goal"
# file = "traces/Llama7B_N4_GPU16_TP1_PP1_DP16_BS32/llama.goal"
# file = "trace_llama7b.goal"

traffic_counter = defaultdict(lambda :0)
total_comp = 0
# sample line for send "send 0b to 2 tag 00000000 cpu 0 nic 0"
send_line_pattern = r''
for line in open(file, "r"):
    if "send" in line:
        size = int(line.split(" ")[2][:-1])
        traffic_counter[size] += 1
    if "calc" in line:
        size = int(line.split(" ")[2])
        total_comp += size

total_traffic = 0
for k in sorted(traffic_counter.keys()):
    total_traffic += k * traffic_counter[k]
    # print(k, traffic_counter[k])

print(f"total traffic: {total_traffic} bytes")
print(f"total computation: {total_comp} ns")