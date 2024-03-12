import onnx
import torch
import onnxruntime as ort

import torchaudio
from torch.nn import functional as F

from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    shape_inference,
    CalibrationDataReader,
)

from torch_df_streaming_minimal import TorchDFMinimalPipeline

onnx.helper.make_sequence_value_info = onnx.helper.make_tensor_sequence_value_info
AUDIO_PATH = "examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav"


class EncoderDataReader(CalibrationDataReader):
    def __init__(self, calibration_audio_path, input_names, n_samples=10):
        self.enum_data = None

        self.data_list = self._preprocess_frames(calibration_audio_path, n_samples)
        self.input_names = input_names
        self.datasize = len(self.data_list)

    def _preprocess_frames(self, calibration_audio_path, n_samples):
        input_audio, _ = torchaudio.load(calibration_audio_path, channels_first=True)
        input_audio = input_audio.mean(dim=0).unsqueeze(0).cpu()

        input_audio = input_audio.squeeze(0)
        orig_len = input_audio.shape[0]

        hop_size_divisible_padding_size = (480 - orig_len % 480) % 480
        orig_len += hop_size_divisible_padding_size
        input_audio = F.pad(input_audio, (0, 960 + hop_size_divisible_padding_size))

        torch_df = TorchDFMinimalPipeline(device="cpu")

        chunked_audio = torch.split(input_audio, 480)
        input_samples = []

        streaming_model = torch_df.torch_streaming_model
        states = torch_df.states
        (
            erb_norm_state,
            band_unit_norm_state,
            analysis_mem,
            synthesis_mem,
            rolling_erb_buf,
            rolling_feat_spec_buf,
            rolling_c0_buf,
            rolling_spec_buf_x,
            rolling_spec_buf_y,
            enc_hidden,
            erb_dec_hidden,
            df_dec_hidden,
        ) = states

        for frame in chunked_audio[:n_samples]:
            new_states = streaming_model(frame, *states)[1:]
            (
                new_erb_norm_state,
                new_band_unit_norm_state,
                new_analysis_mem,
                new_synthesis_mem,
                new_rolling_erb_buf,
                new_rolling_feat_spec_buf,
                new_rolling_c0_buf,
                new_rolling_spec_buf_x,
                new_rolling_spec_buf_y,
                new_enc_hidden,
                new_erb_dec_hidden,
                new_df_dec_hidden,
            ) = new_states

            # e0, e1, e2, e3, emb, c0, _ = streaming_model.enc(
            #     new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden
            # )
            input_samples.append(
                [
                    new_rolling_erb_buf.detach().cpu().numpy(),
                    new_rolling_feat_spec_buf.detach().cpu().numpy(),
                    enc_hidden.detach().cpu().numpy(),
                ]
            )
            enc_hidden = new_enc_hidden

        return input_samples

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [
                    {name: x for x, name in zip(samples, self.input_names)}
                    for samples in self.data_list
                ]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class EncoderWrapper:
    def __init__(self, encoder, onnx_conver=False, input_features_example=None):
        self.encoder = encoder
        self.onnx = onnx_conver
        self.input_names = [
            "new_rolling_erb_buf",
            "new_rolling_feat_spec_buf",
            "enc_hidden",
        ]
        self.output_names = [
            "e0",
            "e1",
            "e2",
            "e3",
            "emb",
            "c0",
            "new_enc_hidden",
        ]

        if self.onnx and input_features_example:
            model_path = "encoder.onnx"

            encoder_scripted = torch.jit.script(self.encoder)
            torch.onnx.export(
                encoder_scripted,
                [x.detach() for x in input_features_example],
                model_path,
                verbose=False,
                input_names=self.input_names,
                output_names=self.output_names,
                opset_version=14,
            )

            onnx.checker.check_model(onnx.load(model_path), full_check=True)

            shape_inference.quant_pre_process(
                model_path,
                model_path,
                skip_symbolic_shape=False,
            )
            # quantize_dynamic(model_path, model_path, weight_type=QuantType.QUInt8)
            # dr = EncoderDataReader(AUDIO_PATH, self.input_names)
            # quantize_static(
            #     model_path,
            #     model_path,
            #     dr,
            #     quant_format=QuantFormat.QDQ,
            #     per_channel=False,
            #     weight_type=QuantType.QInt8,
            #     # op_types_to_quantize=["MatMul"],
            # )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                # ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            sess_options.optimized_model_filepath = model_path
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # sess_options.enable_profiling = True

            self.ort_session = ort.InferenceSession(
                model_path, sess_options, providers=["CPUExecutionProvider"]
            )

    def generate_onnx_features(self, input_features):
        return {
            x: y.detach().cpu().numpy()
            for x, y in zip(self.input_names, input_features)
        }

    def __call__(self, new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden):
        if self.onnx:
            input_features_onnx = self.generate_onnx_features(
                (new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden)
            )
            out = self.ort_session.run(
                self.output_names,
                input_features_onnx,
            )
            # self.ort_session.end_profiling()
            return out

        return self.encoder(new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden)


class ERBDecoderWrapper:
    def __init__(self, decoder, onnx_check=False, input_features_example=None):
        self.decoder = decoder
        self.onnx = onnx_check
        self.input_names = ["emb", "e3", "e2", "e1", "e0", "erb_dec_hidden"]
        self.output_names = ["new_gains", "new_erb_dec_hidden"]

        if self.onnx and input_features_example:
            model_path = "erb_decoder.onnx"

            # decoder_scripted = torch.jit.script(self.decoder)
            torch.onnx.export(
                self.decoder,
                input_features_example,
                model_path,
                verbose=False,
                input_names=self.input_names,
                output_names=self.output_names,
                opset_version=14,
            )

            onnx.checker.check_model(onnx.load(model_path), full_check=True)

            shape_inference.quant_pre_process(
                model_path,
                model_path,
                skip_symbolic_shape=False,
            )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
            sess_options.optimized_model_filepath = model_path
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.ort_session = ort.InferenceSession(
                model_path, sess_options, providers=["CPUExecutionProvider"]
            )

    def generate_onnx_features(self, input_features):
        return {
            x: y.detach().cpu().numpy()
            for x, y in zip(self.input_names, input_features)
        }

    def __call__(self, emb, e3, e2, e1, e0, erb_dec_hidden):
        if self.onnx:
            input_features_onnx = self.generate_onnx_features(
                (emb, e3, e2, e1, e0, erb_dec_hidden)
            )
            return self.ort_session.run(
                self.output_names,
                input_features_onnx,
            )

        return self.decoder(emb, e3, e2, e1, e0, erb_dec_hidden)


class DFDecoderWrapper:
    def __init__(self, decoder, onnx=False, input_features_example=None):
        self.decoder = decoder
        self.onnx = onnx
        self.input_names = ["emb", "new_rolling_c0_buf", "df_dec_hidden"]
        self.output_names = ["new_coefs", "new_df_dec_hidden"]

        if self.onnx and input_features_example:
            model_path = "df_decoder.onnx"

            # decoder_scripted = torch.jit.script(self.decoder)
            torch.onnx.export(
                self.decoder,
                input_features_example,
                model_path,
                verbose=False,
                input_names=self.input_names,
                output_names=self.output_names,
                opset_version=14,
            )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            sess_options.optimized_model_filepath = model_path
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.ort_session = ort.InferenceSession(
                model_path, sess_options, providers=["CPUExecutionProvider"]
            )

    def generate_onnx_features(self, input_features):
        return {
            x: y.detach().cpu().numpy()
            for x, y in zip(self.input_names, input_features)
        }

    def __call__(self, emb, new_rolling_c0_buf, df_dec_hidden):
        if self.onnx:
            input_features_onnx = self.generate_onnx_features(
                (emb, new_rolling_c0_buf, df_dec_hidden)
            )
            return self.ort_session.run(
                self.output_names,
                input_features_onnx,
            )

        return self.decoder(emb, new_rolling_c0_buf, df_dec_hidden)
