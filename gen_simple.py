from __future__ import annotations
from goal_v2 import GoalCalc, GoalSequential, GoalParallel, GoalSend, GoalRecv, GoalOp, GoalRank, GoalCPU, _current_self_rank, _current_cpu
from typing import List, Tuple, Dict, Optional
import contextvars
from contextlib import ContextDecorator
from math import log2
from nccl_primitives import init_data, reduction_time, copy_time
from nccl_comm import NCCLProto
import numpy as np
    
class Communicator:
    def __init__(self, ranks: List[GoalRank]):
        self.ranks: List[GoalRank] = tuple(ranks) # make it immutable
        self.channels: List[Tuple[List[GoalRank], Dict[GoalRank, int]]] = []
        # self.add_channel(list(range(len(ranks))))
    
    def add_channel(self, channel_order: List[int]) -> None:
        assert(set(channel_order) == set(range(len(self.ranks))))
        channel_ranks = [self.ranks[i] for i in channel_order]
        rank2next_rank = {channel_ranks[i]: i for i in range(len(channel_ranks))}
        self.channels.append((channel_ranks, rank2next_rank))
    
    def get_goal_rank(self, rank: int, channel: int = 0) -> GoalRank:
        return self.channels[channel][0][rank]
    
    def get_comm_rank(self, rank: GoalRank, channel: int = 0) -> int:
        return self.channels[channel][1][rank]


def get_reduction_time(size: int) -> int:
    # Dummy implementation, replace with actual model
    return size // (1024)  # Assume 1 MB/ms reduction speed

class CollectiveOp:
    def __init__(self, comm: Communicator, size: int):
        self.comm: Communicator = comm
        self.size: int = size
    
    def to_goal(self) -> GoalOp:
        def goal_gen():
            for channel in range(len(self.comm.channels)):
                with GoalCPU(0, offset_mode=True):
                    yield self._to_goal_chnl(channel)
        return GoalParallel(goal_gen())

    def _to_goal_chnl(self, channel: int) -> GoalOp:
        pass

class AllReduce_v1(CollectiveOp):
    def __init__(self, comm: Communicator, size: int):
        super().__init__(comm, size)
    
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        comm_rounds = int(log2(num_ranks))
        def goal_gen():
            block_size = self.size // num_ranks
            for r in range(comm_rounds):
                peer_rank = self.comm.get_goal_rank(self_comm_rank ^ (1 << r), channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=0)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=0)
                # red_op = GoalCalc(get_reduction_time(block_size))
                red_op = GoalCalc(reduction_time(block_size, 0))
                yield GoalSequential([send_op, GoalSequential([recv_op, red_op])], dependency=False)
                block_size *= 2
        return GoalSequential(goal_gen())

class AllReduce_v2(CollectiveOp):
    def __init__(self, comm: Communicator, size: int):
        super().__init__(comm, size)
    
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])

if __name__ == "__main__":
    # Example usage
    init_data("npkit_benchmark_results/ault/npkit_data_summary_Simple.json", "npkit_benchmark_results/ault/npkit_data_summary_LL.json")
    nranks = 8
    ranks = [GoalRank(i) for i in range(nranks)]
    comms = []
    chnls = np.arange(nranks)
    # for ch in range(int(log2(nranks))):
    #     comm = Communicator(ranks)
    #     comm.add_channel(chnls.tolist())
    #     comms.append(comm)
    #     chnls = chnls.reshape((nranks // 2, 2)).T.flatten()
    # allreduce_ops = [AllReduce_v1(comm, size=1024*1024*1) for comm in comms]  # 1 MB
    # with open("allreduce_simple8.goal", "w") as f:
    #     f.write(f"num_ranks {nranks}\n")
    #     for rank in ranks:
    #         with rank:
    #             goal_op = GoalParallel([allreduce_op.to_goal() for allreduce_op in allreduce_ops], cpu=0)
    #             f.write(f"rank {rank.self_rank} {{\n")
    #             for line in goal_op.generate_lines():
    #                 f.write(line + "\n")
    #             f.write("}\n")
    comm = Communicator(ranks)
    for ch in range(int(log2(nranks))):
        comm.add_channel(chnls.tolist())
        chnls = chnls.reshape((nranks // 2, 2)).T.flatten()
    allreduce_op = AllReduce_v1(comm, size=1024*1024*1)  # 1 MB
    with open("allreduce_simple8.goal", "w") as f:
        f.write(f"num_ranks {nranks}\n")
        for rank in ranks:
            with rank:
                goal_op = allreduce_op.to_goal()
                f.write(f"rank {rank.self_rank} {{\n")
                for line in goal_op.generate_lines():
                    f.write(line + "\n")
                f.write("}\n")