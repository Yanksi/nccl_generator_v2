from __future__ import annotations
from unused_modules.goal_v2 import GoalCalc, GoalSequential, GoalParallel, GoalSend, GoalRecv, GoalOp, GoalRank, GoalCPU, _current_self_rank, _current_cpu
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
    return size // (32)  # Assume 32 B/ns reduction speed

def get_copy_time(size: int) -> int:
    # Dummy implementation, replace with actual model
    # return size // (64)  # Assume 64 B/ns copy speed
    return 0

class CollectiveOp:
    def __init__(self, comm: Communicator, size: int, context: int = 0):
        self.comm: Communicator = comm
        self.size: int = size
        self.context: int = context
    
    def to_goal(self) -> GoalOp:
        def goal_gen():
            for channel in range(len(self.comm.channels)):
                with GoalCPU(0, offset_mode=True):
                    yield self._to_goal_chnl(channel)
        return GoalParallel(goal_gen())

    def _to_goal_chnl(self, channel: int) -> GoalOp:
        pass

class AllReduce_RecDoub(CollectiveOp): # Reduce-Scatter followed by All-Gather
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        comm_rounds = int(log2(num_ranks))
        def goal_gen():
            # Reduce-Scatter phase
            block_size = self.size // 2
            for r in range(comm_rounds):
                peer_rank = self.comm.get_goal_rank(self_comm_rank ^ (1 << r), channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=self.context)
                red_op = GoalCalc(get_reduction_time(block_size))
                yield GoalSequential([send_op, GoalSequential([recv_op, red_op])], dependency=False)
                block_size //= 2
            # All-Gather phase
            block_size *= 2
            for r in range(comm_rounds):
                peer_rank = self.comm.get_goal_rank(self_comm_rank ^ (1 << (comm_rounds - 1 - r)), channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=self.context)
                yield GoalSequential([send_op, recv_op], dependency=False)
                block_size *= 2
        return GoalSequential(goal_gen())


class AllReduce_Ring(CollectiveOp):
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        def goal_gen():
            block_size = self.size // num_ranks
            # Reduce-Scatter phase
            for r in range(num_ranks - 1):
                send_peer = self.comm.get_goal_rank((self_comm_rank + 1) % num_ranks, channel)
                recv_peer = self.comm.get_goal_rank((self_comm_rank - 1 + num_ranks) % num_ranks, channel)
                send_op = GoalSend(send_peer, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(recv_peer, block_size, nic=0, context=self.context)
                red_op = GoalCalc(get_reduction_time(block_size))
                yield GoalSequential([send_op, GoalSequential([recv_op, red_op])], dependency=False)
            # All-Gather phase
            for r in range(num_ranks - 1):
                send_peer = self.comm.get_goal_rank((self_comm_rank + 1) % num_ranks, channel)
                recv_peer = self.comm.get_goal_rank((self_comm_rank - 1 + num_ranks) % num_ranks, channel)
                send_op = GoalSend(send_peer, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(recv_peer, block_size, nic=0, context=self.context)
                yield GoalSequential([send_op, recv_op], dependency=False)
        return GoalSequential(goal_gen())

class ReduceScatter_RecDoub(CollectiveOp):
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        comm_rounds = int(log2(num_ranks))
        def goal_gen():
            block_size = self.size // 2
            for r in range(comm_rounds):
                peer_rank = self.comm.get_goal_rank(self_comm_rank ^ (1 << r), channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=self.context)
                red_op = GoalCalc(get_reduction_time(block_size))
                # red_op = GoalCalc(reduction_time(block_size, 0))
                yield GoalSequential([send_op, GoalSequential([recv_op, red_op])], dependency=False)
                block_size //= 2
        return GoalSequential(goal_gen())


class ReduceScatter_Ring(CollectiveOp):
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        def goal_gen():
            block_size = self.size // num_ranks
            for r in range(num_ranks - 1):
                send_peer = self.comm.get_goal_rank((self_comm_rank + 1) % num_ranks, channel)
                recv_peer = self.comm.get_goal_rank((self_comm_rank - 1 + num_ranks) % num_ranks, channel)
                send_op = GoalSend(send_peer, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(recv_peer, block_size, nic=0, context=self.context)
                red_op = GoalCalc(get_reduction_time(block_size))
                yield GoalSequential([send_op, GoalSequential([recv_op, red_op])], dependency=False)
        return GoalSequential(goal_gen())


class AllGather_RecDoub(CollectiveOp):
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        comm_rounds = int(log2(num_ranks))
        def goal_gen():
            block_size = self.size // num_ranks
            for r in range(comm_rounds):
                peer_rank = self.comm.get_goal_rank(self_comm_rank ^ (1 << (comm_rounds - 1 - r)), channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=self.context)
                yield GoalSequential([send_op, recv_op], dependency=False)
                block_size *= 2
        return GoalSequential(goal_gen())

class AllGather_Ring(CollectiveOp):
    def __init__(self, comm: Communicator, size: int):
        super().__init__(comm, size)
    
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        def goal_gen():
            block_size = self.size // num_ranks
            for r in range(num_ranks - 1):
                send_peer = self.comm.get_goal_rank((self_comm_rank + 1) % num_ranks, channel)
                recv_peer = self.comm.get_goal_rank((self_comm_rank - 1 + num_ranks) % num_ranks, channel)
                send_op = GoalSend(send_peer, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(recv_peer, block_size, nic=0, context=self.context)
                yield GoalSequential([send_op, recv_op], dependency=False)
        return GoalSequential(goal_gen())

class AlltoAll_Linear(CollectiveOp):
    def _to_goal_chnl(self, channel: int) -> GoalOp:
        self_rank = _current_self_rank()
        self_comm_rank = self.comm.get_comm_rank(self_rank, channel)
        num_ranks = len(self.comm.channels[channel][0])
        block_size = self.size // num_ranks
        def goal_gen():
            for r in range(num_ranks):
                if r == self_comm_rank:
                    continue
                peer_rank = self.comm.get_goal_rank(r, channel)
                send_op = GoalSend(peer_rank, block_size, nic=0, context=self.context)
                recv_op = GoalRecv(peer_rank, block_size, nic=0, context=self.context)
                yield GoalSequential([send_op, recv_op], dependency=False)
        return GoalParallel(goal_gen())

def gen_pp_edge():
    nranks = 8
    ranks = [GoalRank(i) for i in range(nranks)]
    comm = Communicator(ranks)
    comm.add_channel(list(range(nranks)))
    chnls = np.arange(nranks)
    all_gather = AllGather_RecDoub(comm, size=1024*1024*8)
    reduce_scatter = ReduceScatter_RecDoub(comm, size=1024*1024*8)
    # all_gather = AllGather_Ring(comm, size=1024*1024*8)
    # reduce_scatter = ReduceScatter_Ring(comm, size=1024*1024*8)
    expected_fwd_comp_time = get_reduction_time(1024 * 1024 * 25)
    expected_bwd_comp_time = get_reduction_time(1024 * 1024 * 25)
    comp_segments = 4
    with open("dp_edge.goal", "w") as f:
        f.write(f"num_ranks {nranks}\n")
        for rank in ranks:
            with rank:
                forward_op = GoalSequential([all_gather.to_goal(), GoalSequential(GoalCalc(expected_fwd_comp_time // comp_segments) for _ in range(comp_segments))])
                backward_op = GoalSequential([GoalSequential(GoalCalc(expected_bwd_comp_time // comp_segments) for _ in range(comp_segments)), reduce_scatter.to_goal()])
                f.write(f"rank {rank.self_rank} {{\n")
                for line in forward_op.generate_lines():
                    f.write(line + "\n")
                for line in backward_op.generate_lines():
                    f.write(line + "\n")
                f.write("}\n")


def gen_mix():
    nranks = 8
    ranks = [GoalRank(i) for i in range(nranks)]
    comm = Communicator(ranks)
    comm.add_channel(list(range(nranks)))
    alltoall = AlltoAll_Linear(comm, size=1024*1024*8, context=1)
    all_gather = AllGather_RecDoub(comm, size=1024*1024*8, context=2)
    reduce_scatter = ReduceScatter_RecDoub(comm, size=1024*1024*8, context=2)
    expected_fwd_comp_time = get_reduction_time(1024 * 1024 * 64)
    expected_bwd_comp_time = get_reduction_time(1024 * 1024 * 64)
    comp_segments = 4
    with open("mix_edge.goal", "w") as f:
        f.write(f"num_ranks {nranks}\n")
        for rank in ranks:
            with rank:
                forward_op = GoalSequential([all_gather.to_goal(), GoalSequential(GoalCalc(expected_fwd_comp_time // comp_segments) for _ in range(comp_segments)), alltoall.to_goal()])
                backward_op = GoalSequential([alltoall.to_goal(), GoalSequential(GoalCalc(expected_bwd_comp_time // comp_segments) for _ in range(comp_segments)), reduce_scatter.to_goal()])
                f.write(f"rank {rank.self_rank} {{\n")
                for line in forward_op.generate_lines():
                    f.write(line + "\n")
                for line in backward_op.generate_lines():
                    f.write(line + "\n")
                f.write("}\n")


def gen_moe():
    nranks = 8
    ranks = [GoalRank(i) for i in range(nranks)]
    comm = Communicator(ranks)
    comm.add_channel(list(range(nranks)))
    alltoall = AlltoAll_Linear(comm, size=1024*1024*8)
    expected_fwd_comp_time = get_reduction_time(1024 * 1024 * 64)
    expected_bwd_comp_time = get_reduction_time(1024 * 1024 * 64)
    comp_segments = 4
    with open("moe_edge.goal", "w") as f:
        f.write(f"num_ranks {nranks}\n")
        for rank in ranks:
            with rank:
                forward_op = GoalSequential([GoalSequential(GoalCalc(expected_fwd_comp_time // comp_segments) for _ in range(comp_segments)), alltoall.to_goal()])
                backward_op = GoalSequential([alltoall.to_goal(), GoalSequential(GoalCalc(expected_bwd_comp_time // comp_segments) for _ in range(comp_segments))])
                f.write(f"rank {rank.self_rank} {{\n")
                for line in forward_op.generate_lines():
                    f.write(line + "\n")
                for line in backward_op.generate_lines():
                    f.write(line + "\n")
                f.write("}\n")

if __name__ == "__main__":
    # Example usage
    # init_data("npkit_benchmark_results/ault/npkit_data_summary_Simple.json", "npkit_benchmark_results/ault/npkit_data_summary_LL.json")
    # nranks = 8
    # ranks = [GoalRank(i) for i in range(nranks)]
    # comms = []
    # chnls = np.arange(nranks)

    # all_reduces = []
    # # comm = Communicator(ranks)
    # for ch in range(int(log2(nranks))):
    #     comm = Communicator(ranks)
    #     comm.add_channel(chnls.tolist())
    #     # chnls = chnls.reshape((nranks // 2, 2)).T.flatten()
    #     all_reduces.append(AllReduce_v2(comm, size=1024*1024*8))
    #     # all_reduces.append(AllReduce_v2(comm, size=1024*1024*8))
    # with open("allreduce_b2b_v2.goal", "w") as f:
    #     f.write(f"num_ranks {nranks}\n")
    #     for rank in ranks:
    #         with rank:
    #             goal_op = GoalSequential((op.to_goal() for op in all_reduces))
    #             f.write(f"rank {rank.self_rank} {{\n")
    #             for line in goal_op.generate_lines():
    #                 f.write(line + "\n")
    #             f.write("}\n")
    gen_pp_edge()
    gen_moe()
    gen_mix()