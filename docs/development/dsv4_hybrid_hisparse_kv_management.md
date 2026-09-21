# DSv4 Hybrid HiSparse KV Management

This document describes the management-only implementation. It does not define
the attention, host-transfer, or decode-graph kernels. Radix cache and
speculative decoding are out of scope for this phase.

## Units and pool

DSv4 C4 uses a paged GPU allocator. A C4 page is the accounting unit shared by
resident KV and a request's hot buffer. `total_c4_blocks` is the allocator
capacity in pages and `free_c4_blocks` is its current available capacity in
pages. Token counts must be converted with `ceil(tokens / c4_page_size)` before
they are compared with a watermark.

For normal decode, the model top-k determines the hot rows. The default is
`device_buffer_size = 2 * index_topk`; the extra factor is the LRU slack used by
vLLM. The per-request hot cost is:

```text
hot_cost_blocks = ceil(device_buffer_size / c4_page_size)
```

If a model has multiple hot groups, sum the block cost of every group.

## Transition policy

The vLLM-compatible transition watermark is:

```text
transition_watermark = max(hot_cost_blocks, floor(total_c4_blocks / 10))
```

At a scheduling boundary, `free_c4_blocks >= transition_watermark` keeps a
request fully resident. Below it, the request is marked for hot transition.
When an allocation reports `missing_blocks`, the reclaim target is:

```text
missing_blocks + max(transition_watermark - free_c4_blocks, 0)
```

The second term restores the reserve after satisfying the immediate
allocation. This policy is implemented as a pure, unit-testable object in
`hisparse_hybrid_policy.py`.

## Eviction sequence

The intended vLLM sequence is page-level:

1. Select an old, unprotected resident page.
2. Ensure its complete host copy is published.
3. Remove its resident mapping and return its GPU page to the shared pool.
4. Allocate a fixed hot region for the request.
5. Keep resident pages, host pages, and hot pages addressable together.

The current SGLang transfer path already copies the full C4 history to host and
`alloc_device_buffer` keeps the newest working-set pages while releasing older
pages. Its request selection remains FIFO and the copy/release operation is
still invoked per request, so it is not yet the full vLLM page scheduler. The
new policy deliberately changes only block accounting and reclaim targets; a
future page-level implementation should add explicit resident-page state and
pending-copy state before changing transfer ordering.

## Verification

Record these values in GPU runs:

* `total_c4_blocks`, `free_c4_blocks`, `hot_cost_blocks`, and
  `transition_watermark`;
* request transitions and reclaim target/actual blocks;
* host-copy completions, released resident pages, and allocated hot pages;
* reclaim latency and the number of mixed resident/host requests.

Low-pressure tests must show no reclaim. A pressure test must show transition
below the watermark, host publication before GPU release, and correct decode
after host top-k misses. Run with `--disable-radix-cache`; decode CUDA Graph is
currently disabled for the DSv4 hybrid path.
