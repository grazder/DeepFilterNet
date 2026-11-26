# TorchDF

Commit against which the comparison was made - https://github.com/Rikorose/DeepFilterNet/commit/ca46bf54afaf8ace3272aaee5931b4317bd6b5f4

### Export to ONNX with uv

From the repository root, install the export environment with uv and Python 3.10.
The export environment installs the published `deepfilterlib` wheel.

```sh
unzip -n models/DeepFilterNet3_torchDF.zip -d models/
PYTHONPATH="$PWD/DeepFilterNet" uv run --locked --project torchDF/export --python 3.10 \
  python torchDF/model_onnx_export.py --web --test --performance \
  --model-base-dir models/DeepFilterNet3_torchDF --output-path denoiser_web.onnx
```

`--web` selects the minimal streaming model with standard ONNX DFT nodes for
analysis and synthesis. Use ONNX Runtime Web with the WASM execution provider.
Use the flattened initial values from `denoiser_web.states.json` and the shapes
from the ONNX input metadata. Initialize other state inputs to zero and feed
each `new_*` output back into its matching input on the next frame. The bundled
TorchDF checkpoint takes 512 samples per frame at 48 kHz. Use `--minimal` in
place of `--web` to export the baseline for comparison.

### Development and tests

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

The following development and test commands use the Poetry environment.

To run tests:
```
poetry run python -m pytest -v
```
We compare this model to existing `enhance` method (which is partly written on Rust) and tract model (which is purely on Rust). All tests are passing, so model is working.

To enhance audio using streaming implementation:
```
poetry run python torch_df_streaming_minimal.py --audio-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav --output-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary_enhanced.wav
```

I also changed the hop_size parameter from 480 to 512 to speed up the stft. And then finetuned 3 epoches to adapt the model to size 512. New fast model can be found in `models/` dir.  
```
../models
├── DeepFilterNet3_torchDF
│   ├── checkpoints/model_123.ckpt.best
│   └── config.ini
├── DeepFilterNet3_torchDF_onnx
│   ├── denoiser_model.onnx
│   ├── denoiser_model.ort
│   └── denoiser_model.required_operators.config
```

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
