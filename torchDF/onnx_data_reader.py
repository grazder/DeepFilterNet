import torch
import onnxruntime
import torchaudio
import numpy as np
from torch.nn import functional as F
from onnxruntime.quantization import CalibrationDataReader


def _preprocess_frames(calibration_audio_path, size_limit=0):
    input_audio, _ = torchaudio.load(calibration_audio_path, channels_first=True)
    input_audio = input_audio.mean(dim=0).unsqueeze(0).cpu()

    input_audio = input_audio.squeeze(0)
    orig_len = input_audio.shape[0]

    hop_size_divisible_padding_size = (480 - orig_len % 480) % 480
    orig_len += hop_size_divisible_padding_size
    input_audio = F.pad(input_audio, (0, 960 + hop_size_divisible_padding_size))

    chunked_audio = torch.split(input_audio, 480)

    if size_limit > 0 and len(chunked_audio) >= size_limit:
        batch_frames = [chunked_audio[i] for i in range(size_limit)]
    else:
        batch_frames = chunked_audio

    data = [x.cpu().numpy() for x in batch_frames]

    return data


class DeepFilterNetDataReader(CalibrationDataReader):
    def __init__(
        self, calibration_audio_path: str, model_path: str, size_limit: int = 10
    ):
        self.enum_data = None

        session = onnxruntime.InferenceSession(model_path, None)

        # Convert image to input data
        self.data_list = _preprocess_frames(calibration_audio_path, size_limit=0)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: sample} for sample in self.data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
