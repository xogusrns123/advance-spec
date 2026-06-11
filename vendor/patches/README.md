# Vendor patches

This directory holds patch files capturing local modifications to vendored
third-party repositories. The vendored sources themselves live under
`vendor/<name>/` and are gitignored; only the patches here are tracked.

## ArcticInference

- Upstream: https://github.com/snowflakedb/ArcticInference.git
- Base commit: `fba641f8ffbaa25f6715140f4dc85692d6cf7465`
  (`Fix shift-parallel CUDA graph capture and dispatch (#258)`)
- Patch: `ArcticInference.patch`

### Reproduce

```sh
cd vendor
git clone https://github.com/snowflakedb/ArcticInference.git
cd ArcticInference
git checkout fba641f8ffbaa25f6715140f4dc85692d6cf7465
git apply ../patches/ArcticInference.patch
```

### Refresh the patch

After editing files under `vendor/ArcticInference/`, regenerate the patch
(includes any new untracked files via `add -N`):

```sh
cd vendor/ArcticInference
git add -N $(git ls-files --others --exclude-standard)
git diff HEAD > ../patches/ArcticInference.patch
git reset -q HEAD
```

### Patch contents (updated 2026-06-11)

- Reversible undo: `SuffixDecodingCache(enable_undo=True)`,
  `extend_active_response` / `pop_active_response` (bit-exact rollback),
  `temporary_extension(req_id, tokens)` context manager — required by
  `hybrid_spec_decoding/suffix_decoding/suffix_tree.py`,
  `simulation/oracle/chain_hybrid_patch.py`, and the per-anchor sim grafting.
- `SuffixDecodingDraft.counts`: raw per-node suffix-tree counts (input for
  Jeffreys shrinkage in calibration).
- NOTE: PyPI `arctic-inference` (<=0.1.2) has NONE of this; the package must
  be rebuilt from patched source (`pip install . --no-deps` after applying).

Verify after install:

```sh
python3 -c "
from arctic_inference.suffix_decoding import SuffixDecodingCache
c = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=10, enable_undo=True)
c.start_request('r', [1,2,3,1,2]); c.add_active_response('r', [3,4,3,4])
d0 = c.speculate('r', [1,2,3])
ctx = c.temporary_extension('r', [9]); ctx.__enter__(); c.speculate('r', [2,3,9]); ctx.__exit__(None,None,None)
d2 = c.speculate('r', [1,2,3])
assert list(d0.token_ids) == list(d2.token_ids) and list(d0.counts) == list(d2.counts)
print('undo + counts OK')"
```
