# NCCL 2.28 Compatibility Fixes for nccl_generator_v2

This document describes the modifications made to the `nccl_generator_v2` repository to support NCCL 2.28 traces collected from Grok 3B model runs on 4 nodes / 16 GPUs.

## Overview

The original `nccl_generator_v2` was designed for NCCL 2.20 traces which have different NVTX marker formats and trace structures. NCCL 2.28 traces required several adaptations:

1. Different trace filename patterns
2. Missing communicator initialization markers
3. Different stream/kernel association patterns
4. Missing data type conversions

## Fixes Applied

### 1. Filename Pattern Extraction (`nsys_events.py`)

**Problem:** NCCL 2.28 traces use filename pattern `profile_<job>_<node>_<rank>.sqlite` instead of the expected `nid<nodeId>_*` pattern.

**Solution:** Added `extract_node_id()` function that supports both formats:

```python
def extract_node_id(filename: str) -> str:
    """Extract node ID from trace filename.
    
    Supports two formats:
    - nid<nodeId>_*.sqlite (original NCCL 2.20 format)
    - profile_<job>_<node>_<rank>.sqlite (NCCL 2.28 format)
    """
    name = pathlib.Path(filename).stem
    # Try original nid pattern first
    if name.startswith("nid"):
        match = re.match(r"nid(\d+)", name)
        if match:
            return match.group(1)
    # Try profile_job_node_rank pattern
    match = re.match(r"profile_\d+_(\d+)_\d+", name)
    if match:
        return match.group(1)
    # Fallback: use the full filename stem
    return name
```

### 2. Communicator Info Synthesis (`nsys_events.py`)

**Problem:** NCCL 2.28 traces lack the initialization NVTX markers that contain communicator setup information:
- Missing: `commHash 0x... commId 0x... rank N nranks M pid P`
- Missing: `commHash 0x... Rings [N] prev->my->next pid P`
- Missing: `commHash 0x... Trees [N] child1/child2/child3->my->parent pid P`

**Solution:** Synthesize communicator info from collective operation calls:

```python
# Extract commHash from collective calls like:
# "ncclAllGather(): commHash 0x78721bb24730676a, stream 0x389b2300, data_size..., pid 824993"
coll_pattern = r"nccl[a-zA-Z]+\(\): commHash (0x[0-9a-f]+),.*pid (\d+)"

# Build comm_info from unique (nodeId, pid, commHash) combinations
# Assign sequential ranks based on sorted (nodeId, pid) order

# Synthesize ring topology with 32 channels (NCCL default max)
for channelId in range(32):
    for rank in range(nRanks):
        prevRank = (rank - 1) % nRanks
        nextRank = (rank + 1) % nRanks
        # Create ring entry

# Synthesize binary tree topology with 32 channels
for channelId in range(32):
    for rank in range(nRanks):
        parentRank = (rank - 1) // 2 if rank > 0 else -1
        child1Rank = 2 * rank + 1 if 2 * rank + 1 < nRanks else -1
        child2Rank = 2 * rank + 2 if 2 * rank + 2 < nRanks else -1
        # Create tree entry
```

### 3. Time-Based Kernel-to-NVTX Association (`nsys_events.py`)

**Problem:** NCCL 2.28 runs kernels on 76+ CUDA streams while NVTX events only track ~5 streams. The original stream fingerprint matching failed completely.

**Solution:** Replaced fingerprint-based matching with time-based overlap matching:

```python
def associate_kernel_to_nvtx(comm_grouped, kernel_events, profiling_interval=None):
    """Associate kernel events to NVTX events using time-based matching.
    
    For NCCL 2.28+ traces, kernel events run on many CUDA streams while NVTX
    events only track a few streams. We use time overlap to associate kernels
    to their corresponding NVTX collective calls.
    """
    # For each kernel, find which NVTX event's time interval contains it
    kernel_associations = _associate_events(
        nvtx_starts,
        nvtx_ends,
        nvtx_event_ids,
        kernel_starts,
    )
```

### 4. Missing Data Type Conversion (`nsys_events.py`)

**Problem:** `lastChunkCount` was missing from the `convert_numeric()` call, leaving it as a string type.

**Solution:** Added `lastChunkCount` to the unsigned columns list:

```python
coll_kernel_data = convert_numeric(coll_kernel_data,
    [],[
        "count",
        "chunkCount",
        "workCount",
        "lastChunkCount",  # Was missing!
        "workOffset",
        "sendbuff",
        "recvbuff",
        "pid",
    ]
)
```

### 5. NumPy Type Casting (`gpu.py`)

**Problem:** NumPy `uint64` values passed to Python `range()` cause `TypeError: 'numpy.float64' object cannot be interpreted as an integer`.

**Solution:** Explicitly cast numpy types to Python `int` when creating dataclass instances:

```python
coll_chnl_infos = self.self_gpu.dfs["coll_kernels"][event_id].apply(
    lambda row: CollChnlInfo(
        count=int(row["count"]),
        chunk_count=int(row["chunkCount"]),
        work_count=int(row["workCount"]),
        last_chunk_count=int(row["lastChunkCount"]),
        work_offset=int(row["workOffset"]),
        send_buff=int(row["sendbuff"]),
        recv_buff=int(row["recvbuff"]),
    ),
    axis=1
)
```

### 6. Duplicate Row Handling (`gpu.py`)

**Problem:** NCCL 2.28 traces have multiple `coll_info` rows per event (one per channel), causing `loc[event_id]` to return a DataFrame instead of a Series.

**Solution:** Take the first row when duplicates exist:

```python
coll_info_data = self.self_gpu.dfs["coll_info"].loc[event_id]
# Handle case where there are multiple rows for the same event_id
if isinstance(coll_info_data, pd.DataFrame):
    coll_info_data = coll_info_data.iloc[0]
```

### 7. Negative Gap Handling (`gpu.py`)

**Problem:** Overlapping NVTX events cause `start - prev_end` to be negative, raising `ValueError: Duration must be non-negative`.

**Solution:** Clamp negative gaps to zero:

```python
if prev_end > 0:
    gap = max(0, start - prev_end)  # Clamp negative gaps to 0
    if gap > 0:
        yield GoalCalc(gap, gpu2goal_rank[self.self_gpu], curr_cpu)
```

### 8. Empty Generator Handling (`goal.py`)

**Problem:** Streams with all collectives returning `None` (event not found in coll_info) cause `StopIteration` when `GoalSequential` tries to get the first element.

**Solution:** Catch `StopIteration` and return early for empty generators:

```python
def generate_lines(self) -> Generator[str]:
    iterator = iter(self.ops)
    try:
        self.starting_op = next(iterator)
    except StopIteration:
        # Empty generator - nothing to generate
        self.consumed = True and self.single_use
        return
    # ... rest of method
```

### 9. Empty Dictionary Access (`main_v2.py`)

**Problem:** `p2p_kernels[gpu_id]` and `coll_info[gpu_id]` raise `KeyError` when the dictionaries are empty (no P2P operations in traces).

**Solution:** Use `.get()` with default empty DataFrame:

```python
coll_info_gpu = coll_info.get(gpu_id, pd.DataFrame())
if not coll_info_gpu.empty:
    coll_info_gpu["algo"] = coll_info_gpu["algo"].map(algo_mapping)
    coll_info_gpu["proto"] = coll_info_gpu["proto"].map(proto_mapping)

p2p_kernel_gpu = p2p_kernels.get(gpu_id, pd.DataFrame())
if not p2p_kernel_gpu.empty:
    # Process P2P kernels...
```

## Validation Results

| Metric | Value |
|--------|-------|
| **Input Traces** | 16 SQLite files from Grok 3B on 4 nodes |
| **Expected Runtime** | ~25.15 seconds |
| **Simulated Runtime** | ~24.0 seconds |
| **Difference** | ~4.6% (within acceptable range) |
| **Generated GOAL file** | 2.25 GB |
| **Binary file** | 1.5 GB |

### LogGOPSim Parameters Used
```bash
LogGOPSim -f output.bin -L 3700 -o 200 -g 5 -O 0 -G 0.04 -S 0
```
- `L=3700`: Latency in nanoseconds
- `o=200`: Overhead in nanoseconds  
- `g=5`: Gap (inverse bandwidth) in nanoseconds per byte
- `G=0.04`: GPU gap in nanoseconds per byte
- `S=0`: Message size threshold

## Usage

```bash
cd /users/btommaso/scratch/nccl_generator_v2_grok

python main_v2.py \
    --trace_dir /path/to/nccl_228_traces \
    --output_dir /tmp/output

# Convert to binary
txt2bin -i /tmp/output/output.goal -o /tmp/output/output.bin

# Run simulation
LogGOPSim -f /tmp/output/output.bin -L 3700 -o 200 -g 5 -O 0 -G 0.04 -S 0
```

## Files Modified

1. `nsys_events.py` - Trace parsing and event extraction
2. `gpu.py` - GPU device and collective construction
3. `goal.py` - GOAL file generation
4. `main_v2.py` - Main entry point

## Known Limitations

1. **Missing Event IDs**: Some collective event IDs are not found in `coll_info` (printed as warnings). These are skipped during GOAL generation.

2. **Synthesized Topology**: Ring and tree topologies are synthesized with simple algorithms rather than extracted from actual NCCL topology negotiation. This may differ from the actual NCCL topology.

3. **Fixed Channel Count**: 32 channels are synthesized for all communicators, which may not match the actual channel count used by NCCL.
