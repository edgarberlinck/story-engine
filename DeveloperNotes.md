# Developer Notes

For now sdxl was able to generate great images in a descent time. Still need to test flux. Stable Diffusion could generate a shit, doesn't mean thet suck, maybe it can do a better job in other tasks. Let's see....

flux_dev is great for characters and comples scenes. sdxl can be used for environments and schenes with lesser details. 

---

## Edit conventions (read before editing .py files) — DO NOT SKIP

### 1. This codebase uses non-standard, non-PEP8 indentation that still parses
Many modules (`core/scene_pipeline.py`, `generators/benchmark_video_generator.py`,
`generators/image_engine.py`, `generators/video_engine.py`, `generators/video_generator.py`,
…) indent function bodies with **3, 5 or 6 spaces** instead of the standard 4, and
comments sometimes share a line's indent. This is intentional/legacy: a 3-space block
**is legal Python** (Python only requires a block to be *consistent with its own
siblings*, not with PEP8). So `if True:\n   print('a')` parses fine even though
`black`/flake8 would reject it.

The trap: mixing two different indents **within the same block** (e.g. a 3-space
comment next to a 4-space statement) raises `IndentationError: unexpected indent`.
The `edit` tool's `oldString`/`newString` matching can silently shift one line's
leading spaces relative to its neighbours, producing exactly this mixed block.

**Rule: after ANY edit to these files, immediately run a syntax check before you
trust it or move on** (this is what catches the mixed-indent error at once):

```bash
.venv/bin/python -c "import ast; ast.parse(open('PATH/TO/FILE.py').read()); print('OK')"
```

If it fails, **do not patch line-by-line** — that is what multiplies the mistake.
Instead rewrite the whole broken function in one shot. To do that cleanly, inspect
the *actual* leading-space count first (the `Read` tool hides it), e.g.:

```bash
.venv/bin/python - <<'PY'
lines=open('PATH/TO/FILE.py').readlines()
for i in range(A, B):
    raw=lines[i]; print(f'{i+1:4}| i={len(raw)-len(raw.lstrip()):3}| {raw.rstrip()!r}')
PY
```

Then replace the entire function with a single, internally-consistent block
(any consistent indent works — clean 4-space is safest — since Python only cares
that *each* block is uniform, not that it matches the rest of the file).

### 2. Verify, don't assume, when the edit keeps failing
If a third consecutive edit to the same file still won't parse, stop fiddling and
fall back to rewriting the whole function (step 1) rather than trying one more
`oldString`/`newString` tweak.

### 3. Video quality enhancer (post-generation) — `generators/video_enhancer.py`
Added to fix "aspect ratio looks off" + "movements distort the video" +
"too fast / low quality" on LTX-Video 0.9.5 i2v output:
- **Root cause of the squished aspect ratio:** the scene is generated 1024×1024
     (1:1) but the video is 704×512 (13:10); the old `_prepare_image` did a blind
      `resize((w,h))`, stretching the conditioning frame. Fixed
      (`video_generator._prepare_image`): matching-aspect → clean resize;
      mismatched → **contain-fit + edge-replicated padding, never stretch.**
- **Widescreen end-to-end:** LTX params are now **1024×576 (true 16:9, ÷32)** and
      `step: 50 → 64` for a little more quality; resolution is threaded through
      `image_engine.generate_scene → scene_pipeline → reference_scene_generator`
      so the *scene* is generated at the same 16:9 as the video (source == video).
- **The enhancer is dependency-free** (numpy/scipy/imageio's bundled ffmpeg):
     temporal smoothing (kills flicker/warp) + light spatial denoise + re-encode to
      a high-quality low-CRF **MKV** (default; MP4/AVI available). It hooks into
      `animate_scene(enhance=True)` and the benchmark. Failures degrade to the raw
       clip. No model is downloaded — an opt-in `super_resolve_video` seam runs only
       if a local SR model is present (per ROADMAP "Future Enhancements").
- **Coverage:** `generators/benchmark_*.py` is excluded from the ≥80% coverage gate
     (see `.coveragerc`), so new tests live in `test_*.py` files (also excluded from
      coverage but must pass).