# Goal Trace Generator For NCCL
This new goal generator provides a more modularized design. Thus user can easily add their own implementation of the colelctive algorithm and generate the corresponding traces.

Also, this new generator provides the funcationality of labeling each p2p communication with in the goal trace by letting user put some important information within the `tag` field of each `send` and `recv`.

## Steps for generate a trace
The dependencies used by this project is managed by `uv`, user may install `uv` first for easily installing all the required packages.
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

When user are trying to generate goal file for very large traces, user might encounter the following error:
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