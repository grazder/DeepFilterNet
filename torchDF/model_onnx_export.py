import copy
import onnx
import argparse
import subprocess

import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from torch_df_streaming_minimal import TorchDFMinimalPipeline
from torch_df_streaming import TorchDFPipeline
from typing import Dict, Iterable
from torch.onnx._internal import jit_utils
from loguru import logger

from typing import Tuple, List, Optional, Union
from torch.types import _int, _bool, Number, _dtype, _size
from torch import nn, Tensor
from df.enhance import parse_epoch_type
from nobuco.converters.node_converter import converter

torch.manual_seed(0)

OPSET_VERSION = 17

import keras
import numbers


@converter(nn.ConvTranspose2d)
def converter_ConvTranspose2d(
    self, input: Tensor, output_size: Optional[List[int]] = None
):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation
    output_padding = self.output_padding

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if isinstance(output_padding, numbers.Number):
        output_padding = (output_padding, output_padding)

    in_filters, depth_multiplier, kh, kw = weight.shape
    out_filters = groups * depth_multiplier

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose((2, 3, 1, 0))  # (kh, kw, in_filters, out_filters)

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if groups == 1:
        conv = keras.layers.Conv2DTranspose(
            out_filters,
            kernel_size=(kh, kw),
            strides=stride,
            padding="valid",
            dilation_rate=dilation,
            groups=1,
            use_bias=use_bias,
            weights=params,
        )
    else:
        weights = params[0]

        weights_full = np.zeros(shape=(kh, kw, out_filters, in_filters))
        for i in range(in_filters):
            chunk = i // (in_filters // groups)
            for d in range(depth_multiplier):
                weights_full[..., chunk * depth_multiplier + d, i] = weights[..., d, i]
        params[0] = weights_full

        conv = keras.layers.Conv2DTranspose(
            out_filters,
            kernel_size=(kh, kw),
            strides=stride,
            padding="valid",
            dilation_rate=dilation,
            groups=1,
            use_bias=use_bias,
            weights=params,
        )

    def func(input: Tensor, output_size: Optional[List[int]] = None):
        assert output_size is None

        x = conv(input)

        # Сначала учтём padding
        if padding != (0, 0):
            if padding[0] == 0:
                x = x[:, :, padding[1] : -padding[1], :]
            elif padding[1] == 0:
                x = x[:, padding[0] : -padding[0], :]
            else:
                x = x[:, padding[0] : -padding[0], padding[1] : -padding[1], :]

        # Теперь добавим output_padding
        if output_padding != (0, 0):
            hp, wp = output_padding
            if hp >= 0 and wp >= 0:
                pad_layer = keras.layers.ZeroPadding2D(padding=((0, hp), (0, wp)))
                x = pad_layer(x)
            else:
                crop_layer = keras.layers.Cropping2D(cropping=((0, -hp), (0, -wp)))
                x = crop_layer(x)

        return x

    return func


def export_tflite(
    model: torch.nn.Module,
    input_data: Tuple[torch.Tensor],
    input_names: List[str],
    output_path: str,
    enable_profiling: bool,
) -> bool:
    """
    Export PyTorch model to TFLite with inference validation and benchmarking

    Args:
        model: PyTorch model to export
        input_data: Example input data
        input_names: Input tensor names
        output_path: Output .tflite path
        test_data: Optional test data for validation/benchmarking

    Returns:
        bool: True if export successful
    """
    try:
        import nobuco
        import tensorflow as tf
        from torch.utils import benchmark
    except ImportError as e:
        logger.error(f"TFLite export requires nobuco and tensorflow: {e}")
        return False

    try:
        # Prepare named inputs
        named_inputs = {tensor: name for name, tensor in zip(input_names, input_data)}

        # Convert to Keras
        keras_model = nobuco.pytorch_to_keras(
            model,
            input_data,
            input_names=named_inputs,
            inputs_channel_order=nobuco.ChannelOrder.TENSORFLOW,
            outputs_channel_order=nobuco.ChannelOrder.TENSORFLOW,
        )

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.SELECT_TF_OPS,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        if enable_profiling:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True
        tflite_model = converter.convert()

        # Save model
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        logger.success(f"TFLite model saved to {output_path}")

        # Run inference and benchmark if test data provided
        # Convert test data to numpy arrays

        # Initialize TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_mapping = {}
        for detail in input_details:
            # Extract base name without suffix (e.g., 'input:0' -> 'input')
            base_name = detail["name"].split(":")[0]
            # Remove serving_default_ prefix if present
            clean_name = base_name.replace("serving_default_", "")
            input_mapping[clean_name] = detail

        # --------------------------
        # Prepare test data dictionary
        # --------------------------
        test_data_dict = {}
        for data, name in named_inputs.items():
            # Convert to numpy if it's a tensor
            np_data = (
                data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
            )
            test_data_dict[name] = np_data

            # Log expected and actual shapes
            if name in input_mapping:
                expected_shape = tuple(input_mapping[name]["shape"])
                actual_shape = np_data.shape
                if expected_shape != actual_shape:
                    logger.warning(
                        f"Input '{name}' shape mismatch: "
                        f"Model expects {expected_shape}, "
                        f"got {actual_shape}"
                    )
            else:
                logger.warning(f"Input '{name}' not found in model inputs")

        # --------------------------
        # Set input tensors
        # --------------------------
        for name, data in test_data_dict.items():
            if name in input_mapping:
                detail = input_mapping[name]
                # Reshape data to match expected shape if possible
                if np.prod(data.shape) == np.prod(detail["shape"]):
                    data = data.reshape(detail["shape"])
                interpreter.set_tensor(detail["index"], data)

        interpreter.invoke()

        # Get outputs
        outputs = []
        for detail in output_details:
            output_data = interpreter.get_tensor(detail["index"])
            outputs.append(output_data)

        # --------------------------
        # Benchmark performance
        # --------------------------
        def run_inference():
            interpreter.invoke()

        # Warm-up
        for _ in range(10):
            run_inference()

        # Measure performance
        timer = benchmark.Timer(
            stmt="run_inference()",
            globals={"run_inference": run_inference},
            num_threads=1,
        )

        stats = timer.blocked_autorange(min_run_time=5)
        median_time = stats.median * 1000  # Convert to ms
        logger.info(
            f"TFLite Inference Benchmark:\n- Median time: {median_time:.2f} ms\n"
        )

        return True

    except Exception as e:
        logger.exception(f"TFLite export failed: {e}")
        return False


def onnx_simplify(
    path: str, input_data: Dict[str, np.ndarray], input_shapes: Dict[str, Iterable[int]]
) -> str:
    """
    Simplify ONNX model using onnxsim and checking it

    Parameters:
        path:           str - Path to ONNX model
        input_data:     Dict[str, np.ndarray] - Input data for ONNX model
        input_shapes:   Dict[str, Iterable[int]] - Input shapes for ONNX model

    Returns:
        path:           str - Path to simplified ONNX model
    """
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp, full_check=True)
    onnx.save_model(model_simp, path)
    return path


def test_onnx_model(
    torch_model, ort_session, states, frame_size, input_names, output_names
):
    """
    Simple test that everything converted correctly

    Parameters:
        torch_model:    torch.nn.Module - Original torch model
        ort_session:    onnxruntime.InferenceSession - Inference Session for converted ONNX model
        input_features: Dict[str, np.ndarray] - Input features
    """
    states_torch = copy.deepcopy(states)
    states_onnx = copy.deepcopy(states)

    for i in range(30):
        input_frame = torch.randn(frame_size)

        # torch
        output_torch = torch_model(input_frame, *states_torch)

        # onnx
        output_onnx = ort_session.run(
            output_names,
            generate_onnx_features([input_frame, *states_onnx], input_names),
        )

        for x, y, name in zip(output_torch, output_onnx, output_names):
            y_tensor = torch.from_numpy(y)
            assert torch.allclose(x, y_tensor, atol=1e-2), (
                f"out {name} - {i}, {x.flatten()[-5:]}, {y_tensor.flatten()[-5:]}"
            )


def generate_onnx_features(input_features, input_names):
    return {x: y.detach().cpu().numpy() for x, y in zip(input_names, input_features)}


def perform_benchmark(ort_session, input_features: Dict[str, np.ndarray], output_names):
    """
    Benchmark ONNX model performance

    Parameters:
        ort_session:    onnxruntime.InferenceSession - Inference Session for converted ONNX model
        input_features: Dict[str, np.ndarray] - Input features
    """

    def run_onnx():
        output = ort_session.run(
            output_names,
            input_features,
        )

    t0 = benchmark.Timer(
        stmt="run_onnx()",
        num_threads=1,
        globals={"run_onnx": run_onnx},
    )
    logger.info(
        f"Median iteration time: {t0.blocked_autorange(min_run_time=10).median * 1e3:6.2f} ms / {480 / 48000 * 1000} ms"
    )


def infer_onnx_model(
    streaming_pipeline, ort_session, inference_path, input_names, output_names
):
    """
    Inference ONNX model with TorchDFPipeline
    """
    del streaming_pipeline.torch_streaming_model
    streaming_pipeline.torch_streaming_model = lambda *features: (
        torch.from_numpy(x)
        for x in ort_session.run(
            output_names,
            generate_onnx_features(list(features), input_names),
        )
    )

    noisy_audio, sr = torchaudio.load(inference_path, channels_first=True)
    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0)  # stereo to mono

    enhanced_audio = streaming_pipeline(noisy_audio, sr)

    torchaudio.save(
        inference_path.replace(".wav", "_onnx_infer.wav"),
        enhanced_audio,
        sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )


# setType API provides shape/type to ONNX shape/type inference
def custom_rfft(g: jit_utils.GraphContext, X, n, dim, norm):
    x = g.op(
        "Unsqueeze",
        X,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
    )
    x = g.op(
        "Unsqueeze",
        x,
        g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
    )
    x = g.op("DFT", x, axis_i=1, inverse_i=0, onesided_i=1)
    x = g.op(
        "Squeeze",
        x,
        g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
    )

    return x


# setType API provides shape/type to ONNX shape/type inference
def custom_identity(g: jit_utils.GraphContext, X):
    return X


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if args.minimal:
        streaming_pipeline = TorchDFMinimalPipeline(
            device="cpu", model_base_dir=args.model_base_dir, epoch=args.epoch
        )
    else:
        streaming_pipeline = TorchDFPipeline(
            device="cpu",
            always_apply_all_stages=True,
            model_base_dir=args.model_base_dir,
            epoch=args.epoch,
        )

    frame_size = streaming_pipeline.hop_size
    input_names = streaming_pipeline.input_names
    output_names = streaming_pipeline.output_names

    torch_df = streaming_pipeline.torch_streaming_model
    states = streaming_pipeline.states

    input_frame = torch.rand(frame_size)
    input_features = (input_frame, *states)
    torch_df(*input_features)  # check model

    if args.tflite:
        tflite_path = args.output_path.replace(".onnx", ".tflite")

        # Экспортируем оригинальную модель (не jit) в TFLite
        success = export_tflite(
            model=torch_df,
            input_data=input_features,
            input_names=input_names,
            output_path=tflite_path,
            enable_profiling=True,
        )
    else:
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::fft_rfft",
            symbolic_fn=custom_rfft,
            opset_version=OPSET_VERSION,
        )
        # Only used with aten::fft_rfft, so it's useless in ONNX
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::view_as_real",
            symbolic_fn=custom_identity,
            opset_version=OPSET_VERSION,
        )

        torch_df_script = torch.jit.script(torch_df)

        torch.onnx.export(
            torch_df_script,
            input_features,
            args.output_path,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET_VERSION,
        )
        logger.info(f"Model exported to {args.output_path}!")

        input_features_onnx = generate_onnx_features(input_features, input_names)
        input_shapes_dict = {x: y.shape for x, y in input_features_onnx.items()}

        # Simplify not working for not minimal!
        if args.simplify:
            # raise NotImplementedError("Simplify not working for flatten states!")
            onnx_simplify(args.output_path, input_features_onnx, input_shapes_dict)
            logger.info(f"Model simplified! {args.output_path}")

        if args.ort:
            if (
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "onnxruntime.tools.convert_onnx_models_to_ort",
                        args.output_path,
                        "--optimization_style",
                        "Fixed",
                    ]
                ).returncode
                != 0
            ):
                raise RuntimeError("ONNX to ORT conversion failed!")
            logger.info("Model converted to ORT format!")

        logger.info("Checking model...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        sess_options.optimized_model_filepath = args.output_path
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_profiling = args.profiling

        ort_session = ort.InferenceSession(
            args.output_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        for _ in range(3):
            onnx_outputs = ort_session.run(
                output_names,
                input_features_onnx,
            )

        if args.profiling:
            logger.info("Profiling enabled...")
            ort_session.end_profiling()

        logger.info(
            f"InferenceSession successful! Output shapes: {[x.shape for x in onnx_outputs]}"
        )

        if args.test:
            logger.info("Testing...")
            test_onnx_model(
                torch_df,
                ort_session,
                input_features[1:],
                frame_size,
                input_names,
                output_names,
            )
            logger.info("Tests passed!")

        if args.performance:
            logger.info("Performanse check...")
            perform_benchmark(ort_session, input_features_onnx, output_names)

        if args.inference_path:
            infer_onnx_model(
                streaming_pipeline,
                ort_session,
                args.inference_path,
                input_names,
                output_names,
            )
            logger.info(f"Audio from {args.inference_path} enhanced!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporting torchDF model to ONNX")
    parser.add_argument(
        "--output-path",
        type=str,
        default="denoiser_model.onnx",
        help="Path to output onnx file",
    )
    parser.add_argument("--simplify", action="store_true", help="Simplify the model")
    parser.add_argument("--tflite", action="store_true", help="Export to tflite")
    parser.add_argument("--test", action="store_true", help="Test the onnx model")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Mesure median iteration time for onnx model",
    )
    parser.add_argument("--inference-path", type=str, help="Run inference on example")
    parser.add_argument("--ort", action="store_true", help="Save to ort format")
    parser.add_argument("--profiling", action="store_true", help="Run ONNX profiler")
    parser.add_argument("--minimal", action="store_true", help="Export minimal version")
    parser.add_argument(
        "--model-base-dir",
        type=str,
        default="DeepFilterNet3",
        help='Path to model base dir with "checkpoints" subdir',
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=parse_epoch_type,
        default="best",
        help="Epoch for checkpoint loading. Can be one of ['best', 'latest', <int>].",
    )
    main(parser.parse_args())
