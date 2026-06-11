# ArcticInference reversible-undo patch (suffix tree extend/pop)

`arctic_inference_undo.patch` adds the reversible suffix-tree state needed by
`hybrid_spec_decoding/suffix_decoding/suffix_tree.py` and
`simulation/oracle/chain_hybrid_patch.py`:

- `SuffixDecodingCache(enable_undo=True)` constructor flag
- `extend_active_response` / `pop_active_response` (bit-exact rollback on
  both local and global trees)
- `temporary_extension(req_id, tokens)` context manager
- C++ side: undo journal in `csrc/suffix_decoding/suffix_tree.{h,cc}` +
  bindings; unit test `tests/unit_tests/test_suffix_tree_pop.py`

PyPI `arctic-inference` (<=0.1.2) does NOT have any of this — the package
must be built from patched source.

## Install (inside the sglang-bench container)

```bash
cd /workspace/vendor
git clone https://github.com/snowflakedb/ArcticInference.git
cd ArcticInference
git checkout fba641f   # patch base: "Fix shift-parallel CUDA graph capture and dispatch (#258)"
git apply /workspace/vendor/patches/arctic_inference_undo.patch
pip install . --no-deps   # builds the C++ extension; needs cmake/pybind11 (preinstalled in image)
```

## Verify

```bash
python3 - <<'PY'
from arctic_inference.suffix_decoding import SuffixDecodingCache
c = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=10, enable_undo=True)
c.start_request("r", [1, 2, 3, 1, 2])
c.add_active_response("r", [3, 4, 3, 4])
d0 = c.speculate("r", [1, 2, 3])
with c.temporary_extension("r", [9]):
    c.speculate("r", [2, 3, 9])
d2 = c.speculate("r", [1, 2, 3])
assert list(d0.token_ids) == list(d2.token_ids), "undo not bit-exact"
print("undo OK")
PY
```
