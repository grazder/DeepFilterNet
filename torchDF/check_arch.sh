export CUDA_VISIBLE_DEVICES=-1
# poetry run python benchmark_nn_blocks.py
poetry run python -m pytest -v --disable-warnings -k test_streaming_torch_with_streaming_torch_minimal
poetry run python model_onnx_export.py --performance --simplify