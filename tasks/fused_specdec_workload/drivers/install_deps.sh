#!/bin/bash
cd /dev/shm/w
. venv/bin/activate
# Install the released package for its dependency closure (including sgl_kernel,
# which a bare source tree cannot provide), then re-pin torch, because this
# install drags it forward and the fused extension needs 2.8.0/cu128.
pip install -q 'sglang[srt]' 2>&1 | tail -3
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
python -c "import IPython, torch; print('deps ok, torch', torch.__version__)"
python -c "import sgl_kernel; print('sgl_kernel ok')" 2>&1 | tail -1
echo DEPSDONE
