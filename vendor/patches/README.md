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
