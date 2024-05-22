# TorchDF

Commit against which the comparison was made - https://github.com/Rikorose/DeepFilterNet/commit/ca46bf54afaf8ace3272aaee5931b4317bd6b5f4

Installation:
```
cd path/to/DeepFilterNet/
pip install maturin poetry poethepoet
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
maturin build --release -m pyDF/Cargo.toml

cd DeepFilterNet
export PYTHONPATH=$PWD

cd ../torchDF
poetry install
poe install-torch-cpu
```

Here is presented offline and streaming implementation of DeepFilterNet3 on pure torch. Streaming model can be fully exported to ONNX using `model_onnx_export.py`.

Every script and test have to run inside poetry enviroment.

To run tests:
```
poetry run python -m pytest -v
```
We compare this model to existing `enhance` method (which is partly written on Rust) and tract model (which is purely on Rust). All tests are passing, so model is working.

To enhance audio using streaming implementation:
```
poetry run python torch_df_streaming_minimal.py --audio-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav --output-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary_enhanced.wav
```

To convert model to onnx and run tests:
```
poetry run python model_onnx_export.py --test --performance --inference-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav --ort --simplify --profiling --minimal
```

I also changed the hop_size parameter from 480 to 512 to speed up the stft. And then finetuned 3 epoches to adapt the model to size 512. New fast model can be found in `models/` dir.  

TODO:
* Issues about split + simplify
* Thinkging of offline method exportability + compatability with streaming functions
* torch.where(..., ..., 0) export issue
* dynamo.export check
* thinking of torchDF naming
* rfft hacks tests
* torch.nonzero thinking
* rfft nn.module
* more static methods
