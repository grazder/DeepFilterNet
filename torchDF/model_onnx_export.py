import copy
import json
import onnx
import argparse
import subprocess
from pathlib import Path
from time import perf_counter

import torch
import torchaudio
import numpy as np
import onnxruntime as ort

from torch_df_streaming_minimal import TorchDFMinimalPipeline
from torch_df_streaming import TorchDFPipeline
from typing import Dict, Iterable
from torch.onnx._internal import jit_utils
from loguru import logger

from df.enhance import parse_epoch_type

torch.manual_seed(0)

OPSET_VERSION = 17


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

        states_torch = output_torch[1:]
        states_onnx = [torch.from_numpy(value) for value in output_onnx[1:]]

        for x, y, name in zip(output_torch, output_onnx, output_names):
            y_tensor = torch.from_numpy(y)
            assert torch.allclose(
                x, y_tensor, atol=1e-2
            ), f"out {name} - {i}, {x.flatten()[-5:]}, {y_tensor.flatten()[-5:]}"


def generate_onnx_features(input_features, input_names):
    return {x: y.detach().cpu().numpy() for x, y in zip(input_names, input_features)}


def perform_benchmark(ort_session, input_features, output_names, budget_ms):
    samples_ms = []
    for index in range(550):
        started = perf_counter()
        ort_session.run(output_names, input_features)
        if index >= 50:
            samples_ms.append((perf_counter() - started) * 1e3)
    samples_ms.sort()
    logger.info(
        f"Frame latency p50: {samples_ms[250]:.2f} ms, "
        f"p95: {samples_ms[475]:.2f} ms, frame budget: {budget_ms:.2f} ms"
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


def custom_ifft(g: jit_utils.GraphContext, x, n, dim, norm):
    # The input already has the real/imaginary dimension required by ONNX.
    batch_axis = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    x = g.op("Unsqueeze", x, batch_axis)
    x = g.op("DFT", x, axis_i=1, inverse_i=1, onesided_i=0)
    return g.op("Squeeze", x, batch_axis)


# setType API provides shape/type to ONNX shape/type inference
def custom_identity(g: jit_utils.GraphContext, X):
    return X


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if args.minimal or args.web:
        streaming_pipeline = TorchDFMinimalPipeline(
            device="cpu",
            model_base_dir=args.model_base_dir,
            epoch=args.epoch,
            web_fft=args.web,
        )
    else:
        streaming_pipeline = TorchDFPipeline(
            device="cpu",
            always_apply_all_stages=True,
            model_base_dir=args.model_base_dir,
            epoch=args.epoch
        )

    frame_size = streaming_pipeline.hop_size
    input_names = streaming_pipeline.input_names
    output_names = streaming_pipeline.output_names

    torch_df = streaming_pipeline.torch_streaming_model
    states = streaming_pipeline.states

    input_frame = torch.rand(frame_size)
    input_features = (input_frame, *states)
    torch_df(*input_features)  # check model

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::fft_rfft",
        symbolic_fn=custom_rfft,
        opset_version=OPSET_VERSION,
    )
    # ONNX DFT already represents complex numbers with a trailing dimension of 2.
    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::view_as_real",
        symbolic_fn=custom_identity,
        opset_version=OPSET_VERSION,
    )

    if args.web:
        torch.onnx.register_custom_op_symbolic(
            "aten::view_as_complex", custom_identity, OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::fft_ifft", custom_ifft, OPSET_VERSION
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

    if args.web:
        state_path = Path(args.output_path).with_suffix(".states.json")
        nonzero_states = {
            name: state.flatten().tolist()
            for name, state in zip(input_names[1:], states)
            if torch.count_nonzero(state).item()
        }
        state_path.write_text(json.dumps(nonzero_states) + "\n", encoding="utf-8")
        logger.info(f"Initial states exported to {state_path}")

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
    # Keep the portable ONNX artifact separate from native ORT graph fusions.
    if not args.web:
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
        logger.info("Measuring inference latency...")
        perform_benchmark(
            ort_session,
            input_features_onnx,
            output_names,
            budget_ms=1000 * frame_size / streaming_pipeline.sample_rate,
        )

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
    parser.add_argument("--test", action="store_true", help="Test the onnx model")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Measure median and p95 frame latency",
    )
    parser.add_argument("--inference-path", type=str, help="Run inference on example")
    parser.add_argument("--ort", action="store_true", help="Save to ort format")
    parser.add_argument("--profiling", action="store_true", help="Run ONNX profiler")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Export the minimal model with standard ONNX DFT and web initial states",
    )
    parser.add_argument("--minimal", action="store_true", help="Export minimal version")
    parser.add_argument("--model-base-dir", type=str, default='DeepFilterNet3', help="Path to model base dir with \"checkpoints\" subdir")
    parser.add_argument("-e", "--epoch", type=parse_epoch_type, default='best', help="Epoch for checkpoint loading. Can be one of ['best', 'latest', <int>].")
    main(parser.parse_args())
