# Goal Trace Generator For NCCL
This new goal generator provides a more modularized design. Thus users can easily add their own implementation of the collective algorithm and generate the corresponding traces.

Also, this new generator provides the functionality of labeling each p2p communication in the goal trace by letting users put important information within the `tag` field of each `send` and `recv`.

## Steps for generating a trace
The dependencies used by this project are managed by `uv`; users may install `uv` first for easily installing all the required packages.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then create the environment and install the packages
```bash
uv sync
```

Example command for generating a trace:
```bash
uv run main.py -i traces/MoE8x8B_N16_GPU64_TP1_PP8_DP8_EP1_7B_BS32/sqlite/ -o traces/MoE8x8B_N16_GPU64_TP1_PP8_DP8_EP1_7B_BS32/goal_merged -s npkit_benchmark_results/ault/npkit_data_summary_Simple.json -l npkit_benchmark_results/ault/npkit_data_summary_LL.json  -p -m -r
```

For details of available flags, user may use `uv run main.py -h` to get it.

Note: the `-p` flag is always recommended to be enabled, as it provides much faster generation speed and is generally more memory efficient.

## NCCL 2.20 and NCCL 2.28 traces

The generator supports the older full-annotation traces, such as NCCL 2.20 traces with communicator init, ring/tree, NVTX stream, and kernel annotations, and newer partial-annotation traces, such as NCCL 2.28 traces where some init or stream annotations may be missing.

For full-annotation traces, kernel events are matched to NVTX events with the original stream-fingerprint logic. If stream matching proves that required annotations are missing, the generator falls back to time-based kernel/NVTX association for that GPU only. This preserves the old behavior for fully annotated traces while still allowing partial-annotation traces to be generated.

If communicator init markers are missing, the generator synthesizes communicator membership from observed communication events. If an NCCL runtime log is available, pass it with `--nccl_log`; otherwise the generator automatically checks the parent directory of the trace directory for `log-*.out`.

Supported sqlite filename formats include both `*nid<node>*sqlite` and NCCL 2.28-style `profile_<job>_<node>_<rank>.sqlite`.

Example with an explicit NCCL log:
```bash
uv run main.py \
  -i traces/Grok/sqlite \
  -o traces/Grok/goal_merged \
  -s npkit_benchmark_results/clariden/npkit_data_summary_Simple.json \
  -l npkit_benchmark_results/clariden/npkit_data_summary_LL.json \
  --nccl_log traces/Grok/log-0.out \
  -m -r
```

## Intermediate collective GOAL

Use `--intermediate_goal` to also emit `Events_Dependency.goal`, a V1-style intermediate GOAL file where NCCL operations remain unexploded as collective lines such as `AllGather`, `AllReduce`, and `ReduceScatter` instead of final `send`/`recv` traffic.

Use `--intermediate_goal_only` to write only that intermediate file and skip final send/recv GOAL generation:
```bash
uv run main.py \
  -i traces/Llama/sqlite \
  -o traces/Llama/output_run_v2 \
  -s npkit_benchmark_results/clariden/npkit_data_summary_Simple.json \
  -l npkit_benchmark_results/clariden/npkit_data_summary_LL.json \
  -m --intermediate_goal_only
```

When users are trying to generate a goal file for very large traces, they might encounter the following error:
```
Traceback (most recent call last):
  ...
  File "/home/shuhao/nccl_generator_v2/nccl_primitives.py", line 538, in _to_goal
    return self._p_to_goal(gpu_id2goal_rank, cpu, nic, intra_send, intra_recv), cpu + 1
    ^^^^^^^^^^^^^^^
  File "/home/shuhao/nccl_generator_v2/nccl_primitives.py", line 754, in _p_to_goal
    send = self.send_goal(
    ^^^^^^^^^^^^^^^
  File "/home/shuhao/nccl_generator_v2/nccl_primitives.py", line 480, in send_goal
    GoalSend(
    ^^^
  File "/home/shuhao/nccl_generator_v2/goal.py", line 117, in __init__
    assert(ndigits(self.message_id) <= MESSAGE_ID_WIDTH)
    ^^^^^^^^^^^^^^^
AssertionError
```
This is caused by the tag width allowance within `goal.py` set by `MESSAGE_ID_WIDTH` and `MESSAGE_TAG_WIDTH`. The `MESSAGE_ID_WIDTH` is used for limiting number of digits allowed to be used for an incremental counter for making sure each `send`/`recv` will get a different final tag if they have matching context tag. The `MESSAGE_TAG_WIDTH` is used for limiting the number of digits allowed for the context tag. For a large trace, it is possible that the auto increment counter used more than the allowed number of digits and cause the error above to happen. User may adjust these two values accordingly to resolve the error. It is recommended that the sum of these two widths does not go over 9 as uint32 is used for this tag field in LogGOPSim.

Context tag is provided in function `prepare_gpu_data` in `main.py` through the following lines:
```python
        comm_num_ids = comm_data_gpu["comm_num_id"].to_numpy(dtype=np.int64)
        comm_data_gpu["comm_op_id"] = comm_data_gpu["collective"].map(comm_op_ids)
        comm_op_id = comm_data_gpu["comm_op_id"].to_numpy(dtype=np.int64)
        comm_seq_ids = comm_data_gpu.groupby(["commId", "comm_op_id"]).cumcount().to_numpy(dtype=np.int64)
        comm_seq_ids[comm_data_gpu["collective"].isin(["Send", "Recv"])] = 0  # reset seq_id for point-to-point ops
        comm_identifier = (np.array(hash_sequnces(comm_num_ids, comm_op_id, comm_seq_ids)) % 1000).astype(np.int64)

        comm_data_gpu["context_label"] = parallelism_vals.to_numpy() + comm_identifier * 100
```
User can adjust these lines to provide different context tags to suit their needs.
