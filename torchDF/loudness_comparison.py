import os
import glob
import torch
import argparse
import torchaudio

from tqdm import tqdm
from df import init_df
from torchaudio import transforms
from torch_df_offline import TorchDF

torch.manual_seed(52)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = "cuda"

    model, _, _ = init_df(config_allow_defaults=True, model_base_dir="DeepFilterNet3")
    model.to(device)
    model.eval()

    torch_offline = TorchDF(
        sr=48000, nb_bands=32, min_nb_freqs=2, hop_size=480, fft_size=960, model=model
    )
    torch_offline = torch_offline.to(device)

    print(f"Reading audio from folder - {args.input_folder}")
    clips = glob.glob(os.path.join(args.input_folder, "*.wav")) + glob.glob(
        os.path.join(args.input_folder, "*.flac")
    )
    assert len(clips) > 0, f"Not wound wav or flac in folder {args.input_folder}"
    print(f"Found {len(clips)} audio in {args.input_folder}")

    noisy_loudness_list = []
    enhanced_loudness_list = []

    sample_rate = 48000
    transform = transforms.Loudness(sample_rate)

    for clip in tqdm(clips):
        noisy_audio, noisy_sample_rate = torchaudio.load(clip, channels_first=True)

        if noisy_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=noisy_sample_rate, new_freq=sample_rate
            )
            noisy_audio = resampler(noisy_audio)

        noisy_audio = noisy_audio.mean(dim=0)
        enhanced_audio = torch_offline(noisy_audio.to(device)) * 1.02

        loudness_noisy = transform(noisy_audio.unsqueeze(0)).item()
        loudness_enhanced = transform(enhanced_audio.unsqueeze(0)).item()

        if loudness_enhanced == loudness_enhanced and loudness_noisy == loudness_noisy:
            noisy_loudness_list.append(loudness_noisy)
            enhanced_loudness_list.append(loudness_enhanced)

        save_path = os.path.join(
            args.output_folder, "denoised_" + os.path.basename(clip) + ".wav"
        )
        torchaudio.save(
            save_path,
            enhanced_audio.data.cpu(),
            48000,
        )

    noisy_loudness_list = torch.tensor(noisy_loudness_list)
    enhanced_loudness_list = torch.tensor(enhanced_loudness_list)

    # torch.save(noisy_loudness_list, "noisy_loudness_list.pt")
    # torch.save(enhanced_loudness_list, "enhanced_loudness_list.pt")

    print("Mean Dif:", (noisy_loudness_list - enhanced_loudness_list).mean())
    print("Max Dif:", (noisy_loudness_list - enhanced_loudness_list).max())
    print("Min Dif:", (noisy_loudness_list - enhanced_loudness_list).min())

    print("Mean Abs Dif:", (noisy_loudness_list - enhanced_loudness_list).abs().mean())
    print("Max Abs Dif:", (noisy_loudness_list - enhanced_loudness_list).abs().max())
    print("Min Abs Dif:", (noisy_loudness_list - enhanced_loudness_list).abs().min())
    print(
        "Mean Ratio noisy / enhanced:",
        (noisy_loudness_list / enhanced_loudness_list).mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder", help="path with folder for inference", required=True
    )
    parser.add_argument("--output-folder", help="path to save enhanced audio")
    main(parser.parse_args())
