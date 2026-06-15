# SPOVNOB runtime environment — source before any gate / pipeline / UI run.
#   source env.sh
# Activates the pinned .venv and sets LD_LIBRARY_PATH so ONNXRuntime's CUDA-12
# provider resolves on this CUDA-13 host (driver 580). See UBUNTU_SETUP_GUIDE §5.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"
NVLIB=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")/nvidia
TORCHLIB=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$NVLIB/cufft/lib:$NVLIB/cuda_runtime/lib:$NVLIB/cublas/lib:$TORCHLIB:${LD_LIBRARY_PATH:-}"
export SPOVNOB_MODEL_STORE="/home/user1/model_store"
