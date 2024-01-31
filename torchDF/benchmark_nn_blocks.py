import torch
import torchaudio
import torch.utils.benchmark as benchmark

from torch.nn import functional as F

from torch_df_streaming import TorchDFPipeline
from nn_blocks import EncoderWrapper, ERBDecoderWrapper, DFDecoderWrapper

DEVICE = "cpu"
AUDIO_PATH = "examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav"


def generate_example_inputs(torch_df):
    noisy_audio, sr = torchaudio.load(AUDIO_PATH, channels_first=True)
    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0).to(DEVICE)

    input_audio = noisy_audio.squeeze(0)
    orig_len = input_audio.shape[0]

    # padding taken from
    # https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L229
    hop_size_divisible_padding_size = (
        torch_df.hop_size - orig_len % torch_df.hop_size
    ) % torch_df.hop_size
    orig_len += hop_size_divisible_padding_size
    input_audio = F.pad(
        input_audio, (0, torch_df.fft_size + hop_size_divisible_padding_size)
    )

    chunked_audio = torch.split(input_audio, torch_df.hop_size)
    states = torch_df.states
    atten_lim_db = torch_df.atten_lim_db

    return chunked_audio[1], states, atten_lim_db


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    torch_df = TorchDFPipeline(device=DEVICE, always_apply_all_stages=True)
    frame, states, atten_lim_db = generate_example_inputs(torch_df)

    streaming_model = torch_df.torch_streaming_model
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
    ) = streaming_model.unpack_states(states)

    _, new_states, _ = streaming_model(frame, states, atten_lim_db)
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
    ) = streaming_model.unpack_states(new_states)

    e0, e1, e2, e3, emb, c0, lsnr, _ = streaming_model.enc(
        new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden
    )

    # Encoder
    encoder = EncoderWrapper(
        streaming_model.enc,
        True,
        (new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden),
    )

    def run_encoder():
        _ = encoder(new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden)

    t0 = benchmark.Timer(
        stmt="run_encoder()",
        num_threads=1,
        globals={"run_encoder": run_encoder},
    )
    print(
        f"Median encoder iteration time: {t0.blocked_autorange(min_run_time=10).median * 1e3:6.2f} ms / {480 / 48000 * 1000} ms"
    )

    # ERB Decoder
    erb_decoder = ERBDecoderWrapper(
        streaming_model.erb_dec,
        True,
        (emb, e3, e2, e1, e0, erb_dec_hidden),
    )

    def run_erb_decoder():
        _ = erb_decoder(emb, e3, e2, e1, e0, erb_dec_hidden)

    t0 = benchmark.Timer(
        stmt="run_erb_decoder()",
        num_threads=1,
        globals={"run_erb_decoder": run_erb_decoder},
    )
    print(
        f"Median erb_decoder iteration time: {t0.blocked_autorange(min_run_time=10).median * 1e3:6.2f} ms / {480 / 48000 * 1000} ms"
    )

    # DF Decoder
    erb_decoder = DFDecoderWrapper(
        streaming_model.df_dec,
        True,
        (emb, new_rolling_c0_buf, df_dec_hidden),
    )

    def run_df_decoder():
        _ = erb_decoder(emb, new_rolling_c0_buf, df_dec_hidden)

    t0 = benchmark.Timer(
        stmt="run_df_decoder()",
        num_threads=1,
        globals={"run_df_decoder": run_df_decoder},
    )
    print(
        f"Median df_decoder iteration time: {t0.blocked_autorange(min_run_time=10).median * 1e3:6.2f} ms / {480 / 48000 * 1000} ms"
    )


if __name__ == "__main__":
    main()
