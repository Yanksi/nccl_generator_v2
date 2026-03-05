"""Pipeline Parallel scheduling strategies.

This module provides schedule generators that determine the order of
forward and backward passes across microbatches in pipeline parallelism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import Token


@dataclass(frozen=True)
class ScheduleStep:
    """A single step in the pipeline schedule."""
    kind: Literal["forward", "backward"]
    microbatch: int
    
    def __repr__(self) -> str:
        return f"{'F' if self.kind == 'forward' else 'B'}{self.microbatch}"


class PPSchedule(ABC):
    """Base class for pipeline parallel schedules.
    
    A schedule generates an ordered list of (forward, backward) steps
    for a given PP stage. The order determines when each microbatch's
    forward/backward pass should execute relative to others.
    
    Usage (generate only - caller handles dependencies):
        schedule = OneFOneBSchedule()
        steps = schedule.generate(num_microbatches=8, pp_stage=2, pp_size=4)
        for step in steps:
            if step.kind == "forward":
                do_forward(step.microbatch)
            else:
                do_backward(step.microbatch)
    
    Usage (execute - scheduler handles dependencies):
        schedule = OneFOneBSchedule()
        final_tok = schedule.execute(
            num_microbatches=8, pp_stage=2, pp_size=4,
            do_forward=my_forward,   # (mb: int, after: Token | None) -> Token
            do_backward=my_backward, # (mb: int, after: Token | None) -> Token
        )
    """
    
    def execute(
        self,
        num_microbatches: int,
        pp_stage: int,
        pp_size: int,
        do_forward: Callable[[int, "Token | None"], "Token"],
        do_backward: Callable[[int, "Token | None"], "Token"],
    ) -> "Token":
        """Execute the schedule, creating graph edges for ordering.
        
        This method calls do_forward/do_backward in schedule order and
        chains their completion tokens to enforce the schedule in the graph.
        
        Args:
            num_microbatches: Total number of microbatches
            pp_stage: This GPU's pipeline stage
            pp_size: Total number of pipeline stages
            do_forward: Callback (microbatch, after_token) -> completion_token.
                        Should build forward pass ops, optionally waiting on after_token.
            do_backward: Callback (microbatch, after_token) -> completion_token.
                         Should build backward pass ops, optionally waiting on after_token.
        
        Returns:
            The completion token of the last step (can be used as dependency root).
        """
        steps = self.generate(num_microbatches, pp_stage, pp_size)
        
        prev_tok: Token | None = None
        for step in steps:
            if step.kind == "forward":
                prev_tok = do_forward(step.microbatch, prev_tok)
            else:
                prev_tok = do_backward(step.microbatch, prev_tok)
        
        return prev_tok  # type: ignore  # guaranteed non-None if steps exist
    
    @abstractmethod
    def generate(
        self, 
        num_microbatches: int, 
        pp_stage: int, 
        pp_size: int,
    ) -> List[ScheduleStep]:
        """Generate the schedule as a list of steps.
        
        Args:
            num_microbatches: Total number of microbatches per iteration
            pp_stage: This GPU's pipeline stage (0 = first, pp_size-1 = last)
            pp_size: Total number of pipeline stages
            
        Returns:
            Ordered list of ScheduleSteps to execute
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this schedule."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GPipeSchedule(PPSchedule):
    """GPipe: all forwards first, then all backwards.
    
    Schedule pattern:
        F0 → F1 → F2 → F3 → B0 → B1 → B2 → B3
    
    Characteristics:
        - Simple to implement
        - Large bubble: (PP-1) × (F+B) time
        - High peak memory: holds all microbatch activations simultaneously
    """
    
    @property
    def name(self) -> str:
        return "gpipe"
    
    def generate(self, num_microbatches: int, pp_stage: int, pp_size: int) -> List[ScheduleStep]:
        steps = []
        # All forwards first
        for mb in range(num_microbatches):
            steps.append(ScheduleStep("forward", mb))
        # Then all backwards
        for mb in range(num_microbatches):
            steps.append(ScheduleStep("backward", mb))
        return steps


class OneFOneBSchedule(PPSchedule):
    """1F1B: interleaved forward and backward with warmup/drain phases.
    
    Reduces peak memory by processing backwards as soon as gradients arrive.
    Each stage processes forwards until the pipeline is full, then alternates
    between forward and backward passes.
    
    Schedule pattern for stage 1 in PP=4, with 8 microbatches:
        Warmup:  F0 → F1             (pp_size - stage - 1 = 2 forwards)
        Steady:  F2,B0 → F3,B1 → F4,B2 → F5,B3 → F6,B4 → F7,B5
        Drain:   B6 → B7
    
    Characteristics:
        - Moderate complexity
        - Smaller bubble: (PP-1) × F time
        - Low peak memory: holds ~PP activations at steady state
    """
    
    @property
    def name(self) -> str:
        return "1f1b"
    
    def generate(self, num_microbatches: int, pp_stage: int, pp_size: int) -> List[ScheduleStep]:
        # Warmup: number of forwards before first backward
        # Earlier stages (closer to input) need more warmup to fill the pipeline
        # Later stages (closer to output) start backward sooner
        warmup = min(pp_size - pp_stage - 1, num_microbatches)
        
        steps = []
        
        # Warmup phase: only forwards
        for mb in range(warmup):
            steps.append(ScheduleStep("forward", mb))
        
        # Steady state: 1 forward, 1 backward (interleaved)
        # We do forward for mb (warmup + i), backward for mb i
        num_steady = min(num_microbatches - warmup, num_microbatches)
        for i in range(num_steady):
            mb_fwd = warmup + i
            mb_bwd = i
            # Forward first (if we still have forwards to do)
            if mb_fwd < num_microbatches:
                steps.append(ScheduleStep("forward", mb_fwd))
            # Then backward
            steps.append(ScheduleStep("backward", mb_bwd))
        
        # Drain phase: remaining backwards (if any)
        # This happens when warmup + num_steady backwards < num_microbatches
        for mb in range(num_steady, num_microbatches):
            steps.append(ScheduleStep("backward", mb))
        
        return steps


class InterleavedOneFOneBSchedule(PPSchedule):
    """Interleaved 1F1B with virtual pipeline stages.
    
    Each physical GPU handles multiple "virtual" stages. This reduces
    bubble size by a factor of num_virtual_stages compared to standard 1F1B.
    
    For example, with PP=4 and num_virtual_stages=2:
        - Physical stage 0 handles virtual stages 0 and 4
        - Physical stage 1 handles virtual stages 1 and 5
        - etc.
    
    The schedule interleaves work across virtual stages to minimize idle time.
    
    Note: This schedule requires the model to be split into (PP × V) chunks
    where V is num_virtual_stages. The caller must handle mapping virtual
    stage IDs to model chunks.
    
    Characteristics:
        - Complex implementation
        - Smallest bubble: (PP-1) × F / V time
        - Low peak memory (similar to 1F1B)
    """
    
    def __init__(self, num_virtual_stages: int = 2):
        if num_virtual_stages < 1:
            raise ValueError("num_virtual_stages must be >= 1")
        self.num_virtual_stages = num_virtual_stages
    
    @property
    def name(self) -> str:
        return f"interleaved_1f1b_v{self.num_virtual_stages}"
    
    def generate(self, num_microbatches: int, pp_stage: int, pp_size: int) -> List[ScheduleStep]:
        # For now, raise NotImplementedError with a helpful message
        # Full implementation requires virtual stage tracking in the graph builder
        raise NotImplementedError(
            f"Interleaved 1F1B with {self.num_virtual_stages} virtual stages "
            "requires model chunking support. Use OneFOneBSchedule instead, "
            "or implement virtual stage handling in your graph builder."
        )
    
    def __repr__(self) -> str:
        return f"InterleavedOneFOneBSchedule(num_virtual_stages={self.num_virtual_stages})"
