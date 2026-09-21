"""Pure block-accounting policy for hybrid HiSparse residency transitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridHiSparsePolicy:
    """vLLM-compatible transition policy for one shared GPU block pool.

    ``hot_cost_blocks`` is the complete hot footprint of one request.  The
    watermark is intentionally expressed in blocks, not tokens, because the
    resident and hot groups share a paged allocator.
    """

    total_blocks: int
    hot_cost_blocks: int
    reserve_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_blocks < 0 or self.hot_cost_blocks < 0:
            raise ValueError("block counts must be non-negative")
        if not 0 <= self.reserve_ratio < 1:
            raise ValueError("reserve_ratio must be in [0, 1)")

    @property
    def transition_watermark(self) -> int:
        return max(
            self.hot_cost_blocks,
            int(self.total_blocks * self.reserve_ratio),
        )

    def is_under_pressure(self, free_blocks: int) -> bool:
        return free_blocks < self.transition_watermark

    def reclaim_target(self, free_blocks: int, requested_blocks: int) -> int:
        """Blocks to release to satisfy an allocation and restore the reserve."""
        if requested_blocks < 0:
            raise ValueError("requested_blocks must be non-negative")
        deficit = max(self.transition_watermark - free_blocks, 0)
        return requested_blocks + deficit
