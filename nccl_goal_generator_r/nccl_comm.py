from typing import List, Dict, Type, Union, Optional
from dataclasses import dataclass
from enum import Enum
from math import ceil
from abc import ABC, abstractmethod
from nccl_primitives import *
from __future__ import annotations


class CollAlgo(Enum):
    TREE = 0
    RING = 1

class NCCLProto(Enum):
    LL = 0
    LL128 = 1
    SIMPLE = 2

@dataclass
class TreeTopoNode:
    parent: GPUDevice
    child1: GPUDevice
    child2: GPUDevice
    child3: GPUDevice

@dataclass
class RingTopoNode:
    prev: GPUDevice
    nxt: GPUDevice

class Communicator:
    comm_id: str = None
    rank2gpu: List[GPUDevice] = None
    gpu2rank: Dict[GPUDevice, int] = None
    tree_topo: Dict[GPUDevice, List[TreeTopoNode]] = None
    ring_topo: Dict[GPUDevice, List[RingTopoNode]] = None
    n_ranks: int = 0
    def __init__(self, comm_id: str, gpus: List[GPUDevice]):
        self.comm_id = comm_id
        self.rank2gpu = gpus
        self.gpu2rank = {gpu: i for i, gpu in enumerate(gpus)}
        self.rank2gpu.append(None) # when query rank -1, return None
        self.tree_topo = {gpu: [] for gpu in gpus}
        self.ring_topo = {gpu: [] for gpu in gpus}
        self.n_ranks = len(gpus)
    
    def add_tree_topo(self, gpu_rank: int, parent_rank, child1_rank, child2_rank, child3_rank):
        gpu = self.rank2gpu[gpu_rank]
        parent = self.rank2gpu[parent_rank]
        child1 = self.rank2gpu[child1_rank]
        child2 = self.rank2gpu[child2_rank]
        child3 = self.rank2gpu[child3_rank]
        self.tree_topo[gpu].append(TreeTopoNode(parent, child1, child2, child3))
    
    def add_ring_topo(self, gpu_rank: int, prev_rank, nxt_rank):
        gpu = self.rank2gpu[gpu_rank]
        prev = self.rank2gpu[prev_rank]
        nxt = self.rank2gpu[nxt_rank]
        self.ring_topo[gpu].append(RingTopoNode(prev, nxt))


class CommOp:
    def to_primitives(self) -> NCCLPrimitiveComm:
        raise NotImplementedError

class Send(CommOp):
    def __init__(self, size: int, target_rank: int, comm: Communicator, chunk_size: int):
        self.size = size
        self.peer_gpu = comm.rank2gpu[target_rank]
        if chunk_size < 0:
            raise ValueError("chunk_size must be >= 0")
        self.chunk_size = chunk_size
        
    def to_primitives(self) -> NCCLPrimitiveComm:
        return NCCLSend(target_gpu=self.peer_gpu, size=self.size).proto_simple(self.chunk_size)


class Recv(CommOp):
    def __init__(self, size: int, source_rank: int, comm: Communicator, chunk_size: int):
        self.size = size
        self.peer_gpu = comm.rank2gpu[source_rank]
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = chunk_size
        
    def to_primitives(self) -> NCCLPrimitiveComm:
        return NCCLRecv(source_gpu=self.peer_gpu, size=self.size).proto_simple(self.chunk_size)


@dataclass
class CollInfo:
    root_rank: int
    red_op: int
    algo: CollAlgo
    proto: NCCLProto
    data_size: int
    type_size: int
    chunk_size: int
    chunk_count: int
    chunk_steps: int
    slice_size: int
    slice_steps: int

@dataclass
class CollChnlInfo:
    n_warps: int
    count: int
    chunk_count: int
    work_count: int
    last_chunk_count: int
    work_offset: int
    send_buff: int
    recv_buff: int


class CollectiveOp(CommOp):
    comm: Communicator
    coll_info: CollInfo
    coll_chnl_infos: List[CollChnlInfo]

    def __init__(self, comm: Communicator, coll_info: CollInfo, coll_chnl_infos: List[CollChnlInfo]):
        self.comm = comm
        self.coll_info = coll_info
        self.coll_chnl_infos = coll_chnl_infos

    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        raise NotImplementedError
    
    def to_primitives_tree_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        raise NotImplementedError

    def to_primitives_tree(self, gpu: GPUDevice) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel()
        for chnl_id in range(len(self.coll_chnl_infos)):
            result.add(self.to_primitives_tree_chnl(gpu, chnl_id))
        return result
    
    def to_primitives_ring(self, gpu: GPUDevice) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel()
        for chnl_id in range(len(self.coll_chnl_infos)):
            result.add(self.to_primitives_ring_chnl(gpu, chnl_id))
        return result
    
    def to_primitives(self, gpu: GPUDevice) -> NCCLPrimitiveComm:
        result = None
        if self.coll_info.algo == CollAlgo.RING:
            result = self.to_primitives_ring(gpu)
        elif self.coll_info.algo == CollAlgo.TREE:
            result = self.to_primitives_tree(gpu)
        else:
            raise ValueError(f"Unknown algo: {self.coll_info.algo}")
        if self.coll_info.proto == NCCLProto.LL:
            result = result.proto_ll()
        elif self.coll_info.proto == NCCLProto.SIMPLE:
            result = result.proto_simple(self.coll_info.chunk_size)
        return result
            

class AllReduce(CollectiveOp):
    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[gpu][chnl_id]

        ring_ix = self.comm.gpu2rank[gpu]
        prev_gpu = ring_topo_node.prev
        nxt_gpu = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        last_chunk_count = coll_chnl_info.last_chunk_count

        loop_count = n_ranks * chunk_count


        result = NCCLPrimitiveParallel()
        for elem_offset in range(0, channel_count, loop_count):
            chunk_comm = NCCLPrimitiveSequantial()
            rem_count = channel_count - elem_offset            
            
            if rem_count < loop_count:
                chunk_count = last_chunk_count
                # chunk_count = ceil(rem_count / n_ranks)
                # align_vec_width = 16 // self.coll_info.type_size
                # chunk_count = ceil(chunk_count / align_vec_width) * align_vec_width
            
            # step 0: Send
            chunk = (ring_ix + n_ranks - 1) % n_ranks
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
            chunk_comm.append(NCCLSend(
                target_gpu=nxt_gpu,
                size=n_elems * self.coll_info.type_size
            ))

            # step k-2 steps: reduce and copy to next GPU
            for j in range(2, n_ranks):
                chunk = (ring_ix + n_ranks - j) % n_ranks
                chunk_offset = chunk * chunk_count
                n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
                chunk_comm.append(NCCLRecvReduceSend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))

            # step k-1: recv the reduced data from prev
            chunk = ring_ix
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
            chunk_comm.append(NCCLRecvReduceCopySend(
                source_gpu=prev_gpu,
                target_gpu=nxt_gpu,
                size=n_elems * self.coll_info.type_size
            ))

            # k-2 steps: copy to next GPU
            for j in range(1, n_ranks - 1):
                chunk = (ring_ix + n_ranks - j) % n_ranks
                chunk_offset = chunk * chunk_count
                n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
                chunk_comm.append(NCCLRecvCopySend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))
            
            chunk = (ring_ix + 1) % n_ranks
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
            chunk_comm.append(NCCLRecv(
                source_gpu=prev_gpu,
                size=n_elems * self.coll_info.type_size
            ))

            result.add(chunk_comm)
        return result
    
    def to_primitives_tree_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        tree_topo_node = comm.tree_topo[gpu][chnl_id]
        children = (tree_topo_node.child1, tree_topo_node.child2, tree_topo_node.child3)
        parent = tree_topo_node.parent
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        grid_offset = coll_chnl_info.work_offset
        channel_count = coll_chnl_info.work_count
        last_chunk_count = coll_chnl_info.last_chunk_count
        # TODO



class AllGather(CollectiveOp):
    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[gpu][chnl_id]
        prev_gpu = ring_topo_node.prev
        nxt_gpu = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        count = coll_chnl_info.count
        send_buff = coll_chnl_info.send_buff
        recv_buff = coll_chnl_info.recv_buff
        ring_ix = self.comm.gpu2rank[gpu]

        result = NCCLPrimitiveParallel()
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            chunk_comm = NCCLPrimitiveSequantial()
            # step 0: send the slice of data to next
            if send_buff == recv_buff + (ring_ix * count): # in place send
                chunk_comm.append(NCCLSend(
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))
            else: # CopySend
                chunk_comm.append(NCCLCopySend(
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))
                
            # step k-2 steps:
            for _ in range(n_ranks - 2):
                chunk_comm.append(NCCLRecvCopySend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))

            # step k-1: recv the slice of data from prev
            chunk_comm.append(NCCLRecv(
                source_gpu=prev_gpu,
                size=n_elems * self.coll_info.type_size
            ))
            result.add(chunk_comm)
        return result


class ReduceScatter(CollectiveOp):
    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[gpu][chnl_id]
        prev_gpu = ring_topo_node.prev
        nxt_gpu = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        
        # result = NCCLPrimitiveSequantial()
        result = NCCLPrimitiveParallel()
        # for each chunk, the following communications will be performed
        for chunk_offset in range(0, channel_count, chunk_count):
            chunk_comm = NCCLPrimitiveSequantial()
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            # step 0: send the slice of data to nxt
            chunk_comm.append(NCCLSend(
                target_gpu=nxt_gpu,
                size=n_elems * self.coll_info.type_size
            ))
            
            # step k-2 steps:
            for _ in range(n_ranks - 2):
                chunk_comm.append(NCCLRecvReduceSend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * self.coll_info.type_size
                ))
            
            # step k-1: recv the reduced data from prev
            chunk_comm.append(NCCLRecvReduceCopy(
                source_gpu=prev_gpu,
                size=n_elems * self.coll_info.type_size
            ))
            result.add(chunk_comm)
        return result


class Reduce(CollectiveOp):
    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        ring_topo_node = comm.ring_topo[gpu][chnl_id]
        prev_gpu = ring_topo_node.prev
        nxt_gpu = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count

        coll_info = self.coll_info
        root_gpu = comm.rank2gpu[coll_info.root_rank]
        type_size = coll_info.type_size

        result = NCCLPrimitiveParallel()
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))

            if gpu == root_gpu: # RecvReduceCopy
                result.add(NCCLRecvReduceCopy(
                    source_gpu=prev_gpu,
                    size=n_elems * type_size
                ))
            elif prev_gpu == root_gpu: # Send
                result.add(NCCLSend(
                    target_gpu=nxt_gpu,
                    size=n_elems * type_size
                ))
            else: # RecvReduceSend
                result.add(NCCLRecvReduceSend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * type_size
                ))
        return result


class Broadcast(CollectiveOp):
    def to_primitives_ring_chnl(self, gpu: GPUDevice, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        ring_topo_node = comm.ring_topo[gpu][chnl_id]
        prev_gpu = ring_topo_node.prev
        nxt_gpu = ring_topo_node.nxt

        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count

        coll_info = self.coll_info
        root_gpu = comm.rank2gpu[coll_info.root_rank]
        type_size = coll_info.type_size

        result = NCCLPrimitiveParallel()
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))

            if gpu == root_gpu: # Send
                result.add(NCCLSend(
                    target_gpu=nxt_gpu,
                    size=n_elems * type_size
                ))
            elif nxt_gpu == root_gpu: # Recv
                result.add(NCCLRecv(
                    source_gpu=prev_gpu,
                    size=n_elems * type_size
                ))
            else: # RecvCopySend
                result.add(NCCLRecvCopySend(
                    source_gpu=prev_gpu,
                    target_gpu=nxt_gpu,
                    size=n_elems * type_size
                ))
        return result