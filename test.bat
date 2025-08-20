if ! python -c "import mpm_pytorch" &>/dev/null; then
    echo "Installing mpm_pytorch in editable mode..."
    cd libs/MPM-PyTorch
    pip install -e .
    cd ../..
fi

python main.py --json examples/gorilla/deform.json