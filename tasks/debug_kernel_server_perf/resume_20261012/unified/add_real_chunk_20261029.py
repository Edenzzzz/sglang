"""Let the generated steps carry a REALISTIC prefill chunk (2026-10-29).

The generator hardcodes `PREFILL_Q = 512` and `PREFILL_KV = 4096`: a 512-token chunk on a 4096-token context. A
four-mode serving step carries a chunk of 2048-16384 tokens over a prefix of tens to hundreds of thousands, because
an agent round appends a few thousand tokens onto a long session. With the toy chunk, essentially all work in an
agent-dominant step is long-KV decode input, which is the regime one wide FA3 call already handles well -- so the
measurement answers a question about a step that has almost no prefill in it.

The chunk's share of the step is the second determinant of the dispatcher's gain, alongside the spread of row KV
lengths, and it was being held fixed at an unrealistic value while only the spread varied. This exposes both:

    SYNTH_PREFILL_Q   tokens in the prefill chunk      (default 512)
    SYNTH_PREFILL_KV  context the chunk attends over   (default 4096)

Defaults unchanged, so every previously generated set is reproducible.

  python add_real_chunk.py <ub dir>

Idempotent.
"""
import sys

path = f"{sys.argv[1]}/gen_synth_meld_20261025.py"
src = open(path).read()
if "SYNTH_PREFILL_Q" in src:
    print("already")
    raise SystemExit(0)

OLD = "HEADS, KV_HEADS, D = 28, 4, 128\nPREFILL_Q, PREFILL_KV = 512, 4096\n"
assert OLD in src, "the geometry constants moved; re-read the generator"
NEW = ("HEADS, KV_HEADS, D = 28, 4, 128\n"
       "# The chunk's share of the step decides as much of the dispatcher's gain as the spread of row KV lengths\n"
       "# does, so it must be settable rather than fixed at a token 512 tokens over 4096.\n"
       "import os as _os\n"
       "PREFILL_Q = int(_os.environ.get('SYNTH_PREFILL_Q', 512))\n"
       "PREFILL_KV = int(_os.environ.get('SYNTH_PREFILL_KV', 4096))\n")
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print("parameterized chunk")
