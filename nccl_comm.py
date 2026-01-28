from __future__ import annotations
from typing import List, Dict, Type, Union, Optional, Generator
from dataclasses import dataclass
from enum import Enum
from math import ceil
from abc import ABC, abstractmethod
from nccl_primitives import *


class CollAlgo(Enum):
    TREE = 0
    RING = 1

class NCCLProto(Enum):
    LL = 0
    LL128 = 1
    SIMPLE = 2

@dataclass
class TreeTopoNode:
    parent: GpuId
    children: List[GpuId]

@dataclass
class RingTopoNode:
    prev: GpuId
    nxt: GpuId

class Communicator:
    def __init__(self, comm_id: str, gpu_ids: List[GpuId]):
        self.comm_id: str = comm_id
        self.rank2gpu_id: List[GpuId] = gpu_ids
        self.gpu_id2rank: Dict[GpuId, int] = {gpu_id: i for i, gpu_id in enumerate(gpu_ids)}
        self.n_ranks = len(gpu_ids)
        self.tree_topo: Dict[GpuId, List[TreeTopoNode]] = {gpu_id: [] for gpu_id in gpu_ids}
        self.ring_topo: Dict[GpuId, List[RingTopoNode]] = {gpu_id: [] for gpu_id in gpu_ids}
        self.rank2gpu_id.append(None) # when query rank -1, return None
    
    def add_tree_topo(self, gpu_rank: int, parent_rank: int, children_ranks: List[int]):
        gpu_id = self.rank2gpu_id[gpu_rank]
        parent = self.rank2gpu_id[parent_rank]
        children = [self.rank2gpu_id[child_rank] for child_rank in children_ranks]
        self.tree_topo[gpu_id].append(TreeTopoNode(parent, children))

    def add_ring_topo(self, gpu_rank: int, prev_rank: int, nxt_rank: int):
        gpu_id = self.rank2gpu_id[gpu_rank]
        prev = self.rank2gpu_id[prev_rank]
        nxt = self.rank2gpu_id[nxt_rank]
        self.ring_topo[gpu_id].append(RingTopoNode(prev, nxt))
    
    def __repr__(self) -> str:
        return f"Communicator(id={self.comm_id}, n_ranks={self.n_ranks})"


class CommOp(ABC):
    def __init__(self, gpu_id: GpuId, context: int):
        self.gpu_id = gpu_id
        self.context = context

    @abstractmethod
    def to_primitives(self) -> NCCLPrimitiveComm:
        pass

class CommGrouped(CommOp):
    def __init__(self, gpu_id: GpuId, comm_ops: Union[List[CommOp], Generator[CommOp]], context: int):
        super().__init__(gpu_id, context)
        self.comm_ops = comm_ops
    
    def to_primitives(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel(self.gpu_id, False, (comm_op.to_primitives() for comm_op in self.comm_ops))
        return result

class AllToAll(CommOp):
    def __init__(self, gpu_id: GpuId, size_per_peer: int, comm: Communicator, chunk_size: int, context: int):
        super().__init__(gpu_id, context)
        self.size_per_peer = size_per_peer
        self.comm = comm
        self.chunk_size = chunk_size
    
    def to_primitives(self) -> NCCLPrimitiveComm:
        n_ranks = self.comm.n_ranks
        def generator():
            for peer_rank in range(self.comm.gpu_id2rank[self.gpu_id]):
                # Recv from peer
                yield NCCLRecv(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=self.comm.rank2gpu_id[peer_rank],
                    size=self.size_per_peer,
                    chunk_size=self.chunk_size
                ).proto_simple()
                yield NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=self.comm.rank2gpu_id[peer_rank],
                    size=self.size_per_peer,
                    chunk_size=self.chunk_size
                ).proto_simple()
            for peer_rank in range(self.comm.gpu_id2rank[self.gpu_id] + 1, n_ranks):
                # Send to peer
                yield NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=self.comm.rank2gpu_id[peer_rank],
                    size=self.size_per_peer,
                    chunk_size=self.chunk_size
                ).proto_simple()
                yield NCCLRecv(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=self.comm.rank2gpu_id[peer_rank],
                    size=self.size_per_peer,
                    chunk_size=self.chunk_size
                ).proto_simple()
    
        result = NCCLPrimitiveParallel(self.gpu_id, True, generator())
        return result

@dataclass
class P2PChnlInfo:
    Bytes: int
    proto: NCCLProto
    count: int
    chunk_size: int
    peer_rank: int

class P2POp(CommOp, ABC):
    def __init__(self, gpu_id: GpuId, comm: Communicator, chnl_infos: Union[List[P2PChnlInfo], Generator[P2PChnlInfo]], context: int):
        super().__init__(gpu_id, context)
        self.comm = comm
        # self.peer_gpu_id = comm.rank2gpu_id[peer_rank]
        self.chnl_infos = chnl_infos
    
    def to_primitives(self) -> NCCLPrimitiveComm:
        if not isinstance(self.chnl_infos, list):
            self.chnl_infos = list(self.chnl_infos) # convert generator to list
        if len(self.chnl_infos) == 1:
            return self._to_primitives_chnl(self.chnl_infos[0])
        else:
            result = NCCLPrimitiveParallel(self.gpu_id)
            for chnl_info in self.chnl_infos:
                result.add(self._to_primitives_chnl(chnl_info))
            return result

    def _to_primitives_chnl(self, chnl_info: P2PChnlInfo) -> NCCLPrimitiveComm:
        raise NotImplementedError()

class Send(P2POp):
    def _to_primitives_chnl(self, chnl_info: P2PChnlInfo) -> NCCLPrimitiveComm:
        result = NCCLSend(self.context, self.gpu_id, target_gpu_id=self.comm.rank2gpu_id[chnl_info.peer_rank], size=chnl_info.count, chunk_size=chnl_info.chunk_size)
        if chnl_info.proto == NCCLProto.LL:
            return result.proto_ll()
        elif chnl_info.proto == NCCLProto.LL128:
            return result.proto_ll128()
        elif chnl_info.proto == NCCLProto.SIMPLE:
            return result.proto_simple()
        else:
            raise ValueError(f"Unsupported proto: {chnl_info.proto}")
    # def to_primitives(self) -> NCCLPrimitiveComm:
    #     return NCCLSend(self.context, self.gpu_id, target_gpu_id=self.comm.rank2gpu_id[self.peer_rank], size=self.size, chunk_size=self.chunk_size).proto_simple()


class Recv(P2POp):
    def _to_primitives_chnl(self, chnl_info: P2PChnlInfo) -> NCCLPrimitiveComm:
        result = NCCLRecv(self.context, self.gpu_id, source_gpu_id=self.comm.rank2gpu_id[chnl_info.peer_rank], size=chnl_info.count, chunk_size=chnl_info.chunk_size)
        if chnl_info.proto == NCCLProto.LL:
            return result.proto_ll()
        elif chnl_info.proto == NCCLProto.LL128:
            return result.proto_ll128()
        elif chnl_info.proto == NCCLProto.SIMPLE:
            return result.proto_simple()
        else:
            raise ValueError(f"Unsupported proto: {chnl_info.proto}")

@dataclass
class CollInfo:
    root_rank: int
    red_op: int
    algo: CollAlgo
    proto: NCCLProto
    data_size: int
    type_size: int
    # chunk_size: int
    # chunk_count: int
    chunk_steps: int
    slice_steps: int
    step_size: int

@dataclass
class CollChnlInfo:
    # n_warps: int
    count: int
    chunk_count: int
    work_count: int
    last_chunk_count: int
    work_offset: int
    send_buff: int
    recv_buff: int


class CollectiveOp(CommOp):
    def __init__(self, gpu_id: GpuId, comm: Communicator, coll_info: CollInfo, coll_chnl_infos: Union[List[CollChnlInfo], Generator[CollChnlInfo]], context: int):
        super().__init__(gpu_id, context)
        self.comm = comm
        self.coll_info = coll_info
        self.coll_chnl_infos = coll_chnl_infos

    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        raise NotImplementedError()

    def _to_primitives_tree_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        raise NotImplementedError()

    def _to_primitives_tree(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel(self.gpu_id)
        for chnl_id in range(len(self.coll_chnl_infos)):
            result.add(self._to_primitives_tree_chnl(chnl_id))
        return result
    
    def _to_primitives_ring(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel(self.gpu_id)
        for chnl_id in range(len(self.coll_chnl_infos)):
            result.add(self._to_primitives_ring_chnl(chnl_id))
        return result
    
    def to_primitives(self) -> NCCLPrimitiveComm:
        if not isinstance(self.coll_chnl_infos, list):
            self.coll_chnl_infos = list(self.coll_chnl_infos) # convert generator to list
        result = None
        if self.coll_info.algo == CollAlgo.RING:
            result = self._to_primitives_ring()
        elif self.coll_info.algo == CollAlgo.TREE:
            result = self._to_primitives_tree()
        else:
            raise ValueError(f"Unsupported algo: {self.coll_info.algo}")
        if self.coll_info.proto == NCCLProto.LL: # TODO: distinguish LL and LL128
            result = result.proto_ll()
        elif self.coll_info.proto == NCCLProto.LL128:
            result = result.proto_ll128()
        elif self.coll_info.proto == NCCLProto.SIMPLE:
            result = result.proto_simple()
        else:
            raise ValueError(f"Unsupported proto: {self.coll_info.proto}")
        return result
            

class AllReduce(CollectiveOp):
    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[self.gpu_id][chnl_id]

        ring_ix = self.comm.gpu_id2rank[self.gpu_id]
        prev_gpu_id = ring_topo_node.prev
        nxt_gpu_id = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        last_chunk_count = coll_chnl_info.last_chunk_count

        loop_count = n_ranks * chunk_count

        slice_size = self.coll_info.slice_steps * self.coll_info.step_size
        slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps

        chunk_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        for elem_offset in range(0, channel_count, loop_count):
            rem_count = channel_count - elem_offset            
            
            if rem_count < loop_count:
                chunk_count = last_chunk_count
            
            # step 0: Send
            chunk = (ring_ix + n_ranks - 1) % n_ranks
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
                
            chunk_comm.add(NCCLSend(
                self.context,
                self.gpu_id,
                target_gpu_id=nxt_gpu_id,
                size=n_elems * self.coll_info.type_size,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))

            # step k-2 steps: reduce and copy to next GPU
            for j in range(2, n_ranks):
                chunk = (ring_ix + n_ranks - j) % n_ranks
                chunk_offset = chunk * chunk_count
                n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
                slice_size = max(ceil(n_elems / 16 * slice_per_chunk) * 16, slice_size // 32)
                chunk_comm.add(NCCLRecvReduceSend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))

            # step k-1: recv the reduced data from prev
            chunk = ring_ix
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
            slice_size = max(ceil(n_elems / 16 * slice_per_chunk) * 16, slice_size // 32)
            chunk_comm.add(NCCLRecvReduceCopySend(
                self.context,
                self.gpu_id,
                source_gpu_id=prev_gpu_id,
                target_gpu_id=nxt_gpu_id,
                size=n_elems * self.coll_info.type_size,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))

            # k-2 steps: copy to next GPU
            for j in range(1, n_ranks - 1):
                chunk = (ring_ix + n_ranks - j) % n_ranks
                chunk_offset = chunk * chunk_count
                n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
                slice_size = max(ceil(n_elems / 16 * slice_per_chunk) * 16, slice_size // 32)
                chunk_comm.add(NCCLRecvCopySend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            
            chunk = (ring_ix + 1) % n_ranks
            chunk_offset = chunk * chunk_count
            n_elems = min(chunk_count, max(0, rem_count - chunk_offset))
            slice_size = max(ceil(n_elems / 16 * slice_per_chunk) * 16, slice_size // 32)
            chunk_comm.add(NCCLRecv(
                self.context,
                self.gpu_id,
                source_gpu_id=prev_gpu_id,
                size=n_elems * self.coll_info.type_size,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))

            # result.add(chunk_comm)
        return chunk_comm
    
    def _to_primitives_tree_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        tree_topo_node = comm.tree_topo[self.gpu_id][chnl_id]
        children_gpu_ids = tree_topo_node.children
        parent_gpu_id = tree_topo_node.parent
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count

        result = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            slice_size = self.coll_info.slice_steps * self.coll_info.step_size
            slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps
            # slice_size = max(ceil(n_elems / 16 * slice_per_chunk) * 16, slice_size // 32)
            chunk_comm = NCCLPrimitiveSequential(self.gpu_id)

            if parent_gpu_id is None and len(children_gpu_ids) > 0: # root node
                reduction_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
                for child_gpu_id in children_gpu_ids[:-1]:
                    reduction_comm.add(NCCLRecvReduce(
                        self.context,
                        self.gpu_id,
                        source_gpu_id=child_gpu_id,
                        size=n_elems * self.coll_info.type_size,
                        chunk_size=slice_size,
                        slice_per_chunk=slice_per_chunk
                    ))
                reduction_comm.add(NCCLRecvReduceCopy(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=children_gpu_ids[-1],
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
                chunk_comm.append(reduction_comm)

                broadcast_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
                for child_gpu_id in children_gpu_ids:
                    broadcast_comm.add(NCCLSend(
                        self.context,
                        self.gpu_id,
                        target_gpu_id=child_gpu_id,
                        size=n_elems * self.coll_info.type_size,
                        chunk_size=slice_size,
                        slice_per_chunk=slice_per_chunk
                    ))
                chunk_comm.append(broadcast_comm)
            
            elif parent_gpu_id is not None and len(children_gpu_ids) == 0: # leaf node
                # step 0: send the data to parent
                chunk_comm.append(NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=parent_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
                # step 1: recv the reduced data from parent
                chunk_comm.append(NCCLRecv(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=parent_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            
            elif parent_gpu_id is not None and len(children_gpu_ids) > 0: # internal node
                reduction_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
                for child_gpu_id in children_gpu_ids:
                    reduction_comm.add(NCCLRecvReduce(
                        self.context,
                        self.gpu_id,
                        source_gpu_id=child_gpu_id,
                        size=n_elems * self.coll_info.type_size,
                        chunk_size=slice_size,
                        slice_per_chunk=slice_per_chunk
                    ))
                chunk_comm.append(reduction_comm)
                chunk_comm.append(NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=parent_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
                chunk_comm.append(NCCLRecv(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=parent_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
                
                broadcast_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
                for child_gpu_id in children_gpu_ids:
                    broadcast_comm.add(NCCLSend(
                        self.context,
                        self.gpu_id,
                        target_gpu_id=child_gpu_id,
                        size=n_elems * self.coll_info.type_size,
                        chunk_size=slice_size,
                        slice_per_chunk=slice_per_chunk
                    ))
                chunk_comm.append(broadcast_comm)
            result.add(chunk_comm)
        return result


class AllGather(CollectiveOp):
    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[self.gpu_id][chnl_id]
        prev_gpu_id = ring_topo_node.prev
        nxt_gpu_id = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        count = coll_chnl_info.count
        send_buff = coll_chnl_info.send_buff
        recv_buff = coll_chnl_info.recv_buff
        ring_ix = self.comm.gpu_id2rank[self.gpu_id]

        chunk_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            slice_size = self.coll_info.slice_steps * self.coll_info.step_size
            slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps
            # step 0: send the slice of data to next
            if send_buff == recv_buff + (ring_ix * count): # in place send
                chunk_comm.add(NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            else: # CopySend
                chunk_comm.add(NCCLCopySend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
                
            # step k-2 steps:
            for _ in range(n_ranks - 2):
                chunk_comm.add(NCCLRecvCopySend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))

            # step k-1: recv the slice of data from prev
            chunk_comm.add(NCCLRecv(
                self.context,
                self.gpu_id,
                source_gpu_id=prev_gpu_id,
                size=n_elems,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))
            # result.add(chunk_comm)
        return chunk_comm


class ReduceScatter(CollectiveOp):
    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        n_ranks = comm.n_ranks
        ring_topo_node = comm.ring_topo[self.gpu_id][chnl_id]
        prev_gpu_id = ring_topo_node.prev
        nxt_gpu_id = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count
        
        chunk_comm = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        # for each chunk, the following communications will be performed
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            slice_size = self.coll_info.slice_steps * self.coll_info.step_size
            slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps
            
            # step 0: send the slice of data to nxt
            chunk_comm.add(NCCLSend(
                self.context,
                self.gpu_id,
                target_gpu_id=nxt_gpu_id,
                size=n_elems * self.coll_info.type_size,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))
            
            # step k-2 steps:
            for _ in range(n_ranks - 2):
                chunk_comm.add(NCCLRecvReduceSend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems * self.coll_info.type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            
            # step k-1: recv the reduced data from prev
            chunk_comm.add(NCCLRecvReduceCopy(
                self.context,
                self.gpu_id,
                source_gpu_id=prev_gpu_id,
                size=n_elems * self.coll_info.type_size,
                chunk_size=slice_size,
                slice_per_chunk=slice_per_chunk
            ))
            # result.add(chunk_comm)
        return chunk_comm


class Reduce(CollectiveOp):
    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        ring_topo_node = comm.ring_topo[self.gpu_id][chnl_id]
        prev_gpu_id = ring_topo_node.prev
        nxt_gpu_id = ring_topo_node.nxt
        
        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count

        coll_info = self.coll_info
        root_gpu_id = comm.rank2gpu_id[coll_info.root_rank]
        type_size = coll_info.type_size

        
        slice_size = self.coll_info.slice_steps * self.coll_info.step_size
        slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps

        result = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            
            if self.gpu_id == root_gpu_id: # RecvReduceCopy
                result.add(NCCLRecvReduceCopy(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    size=n_elems * type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            elif prev_gpu_id == root_gpu_id: # Send
                result.add(NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems * type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            else: # RecvReduceSend
                result.add(NCCLRecvReduceSend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems * type_size,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
        return result


class Broadcast(CollectiveOp):
    def _to_primitives_ring_chnl(self, chnl_id: int) -> NCCLPrimitiveComm:
        comm = self.comm
        ring_topo_node = comm.ring_topo[self.gpu_id][chnl_id]
        prev_gpu_id = ring_topo_node.prev
        nxt_gpu_id = ring_topo_node.nxt

        coll_chnl_info = self.coll_chnl_infos[chnl_id]
        chunk_count = coll_chnl_info.chunk_count
        channel_count = coll_chnl_info.work_count

        coll_info = self.coll_info
        root_gpu_id = comm.rank2gpu_id[coll_info.root_rank]
        type_size = coll_info.type_size

        result = NCCLPrimitiveParallel(self.gpu_id, single_executer=True)
        for chunk_offset in range(0, channel_count, chunk_count):
            n_elems = min(chunk_count, max(0, channel_count - chunk_offset))
            slice_size = self.coll_info.slice_steps * self.coll_info.step_size
            slice_per_chunk = self.coll_info.chunk_steps // self.coll_info.slice_steps

            if self.gpu_id == root_gpu_id: # Send
                result.add(NCCLSend(
                    self.context,
                    self.gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            elif nxt_gpu_id == root_gpu_id: # Recv
                result.add(NCCLRecv(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
            else: # RecvCopySend
                result.add(NCCLRecvCopySend(
                    self.context,
                    self.gpu_id,
                    source_gpu_id=prev_gpu_id,
                    target_gpu_id=nxt_gpu_id,
                    size=n_elems,
                    chunk_size=slice_size,
                    slice_per_chunk=slice_per_chunk
                ))
        return result