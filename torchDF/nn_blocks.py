import torch
import onnxruntime as ort


class EncoderWrapper:
    def __init__(self, encoder, onnx=False, input_features_example=None):
        self.encoder = encoder
        self.onnx = onnx
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
            "lsnr",
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

    def __call__(self, new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden):
        if self.onnx:
            input_features_onnx = self.generate_onnx_features(
                (new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden)
            )
            return self.ort_session.run(
                self.output_names,
                input_features_onnx,
            )

        return self.encoder(new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden)


class ERBDecoderWrapper:
    def __init__(self, decoder, onnx=False, input_features_example=None):
        self.decoder = decoder
        self.onnx = onnx
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
