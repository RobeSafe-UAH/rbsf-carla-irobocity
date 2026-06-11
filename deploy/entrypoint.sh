#!/bin/bash



check_venv() {
    echo "🔄 Checking virtual environment..."
    if [ ! -d ".venv" ]; then
        echo " .venv not found — updating virtual environment..."

        if ! uv sync --link-mode=copy; then
            echo "❌ uv sync failed, cleaning up..."
            rm -rf .venv
            exit 1
        fi

        echo "✅ .venv updated"
    else
        echo "✅ .venv found"
    fi
}

check_cuda() {
    echo -e "\n🔍 Checking GPU and CUDA availability..."

    if ! uv run python -c "import torch" 2>/dev/null; then
        echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
        echo -e "   Please check your PyTorch installation!"
    else
        CUDA_AVAILABLE=$(uv run python -c "import torch; print(torch.cuda.is_available())")

        if [ "$CUDA_AVAILABLE" == "True" ]; then
            echo -e "✅ \033[92m\033[1mPyTorch is working properly with the GPU.\033[0m"
            echo -e "📍 GPU Information:"
            uv run python -c "import torch; print(f'   - CUDA version:     {torch.version.cuda}')"
            uv run python -c "import torch; print(f'   - Device name:      {torch.cuda.get_device_name(0)}')"
            uv run python -c "import torch; print(f'   - Number of GPUs:   {torch.cuda.device_count()}')"
        else
            echo -e "❌ \033[91m\033[1mCUDA is not available!\033[0m"
            echo -e "   Check your PyTorch installation"
        fi
    fi
}

# Check if colcon has been executed before, if not, run it to build the workspace
check_colcon(){
    if [ ! -f "install/setup.bash" ]; then
        echo "🚀 Building workspace with colcon..."
        colcon build
    else
        echo "✅ Workspace already built"
    fi
}

###############################################################################

# ========================
# Environment
# ========================

export CARLA_ROOT=/workspace/CARLA
export WORK_DIR=/workspace
export HF_HOME=${WORK_DIR}/weights/hf


export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"
export PYTHONPATH="${WORK_DIR}/.venv/lib/python3.10/site-packages:${PYTHONPATH}"
export PYTHONPATH=/workspace/.venv/lib/python3.10/site-packages:$PYTHONPATH
export PATH="${HOME}/.local/bin:${PATH}"

figlet -c "Robocity"

check_venv
check_colcon

echo -e "\n------------------------------------ System info ----------------------------------------\n"

check_cuda


echo -e "\n-----------------------------------------------------------------------------------------\n"
