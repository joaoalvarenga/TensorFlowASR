import io
import os
import argparse

import tensorflow as tf
import librosa, librosa.display
import numpy as np
import soundfile as sf


parser = argparse.ArgumentParser(prog="Conformer audio file streaming")
parser.add_argument('filename', metavar='FILENAME',
                    help='audio file to be played back')

parser.add_argument("--tflite", type=str, default=None,
                    help="Path to conformer tflite")

args = parser.parse_args()

mls_model = tf.lite.Interpreter(model_path=args.tflite)

def read_raw_audio(audio, sample_rate=16000, force_soundfile=False):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, io.BytesIO):
        if force_soundfile:
            wave, sr = sf.read(audio)
            wave = np.asfortranarray(wave)
            if sr != sample_rate:
                wave = librosa.resample(wave, sr, sample_rate)
        else:
            audio.name = 'out.wav'
            wave, _ = librosa.load(audio, sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave

def recognize(tflitemodel, signal):
    # signal = read_raw_audio(filename)
    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()
    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(
        input_details[1]["index"],
        tf.constant(0, dtype=tf.int32)
    )
    tflitemodel.set_tensor(
        input_details[2]["index"],
        tf.zeros([1, 2, 1, 320], dtype=tf.float32)
    )
    tflitemodel.invoke()
    hyp = tflitemodel.get_tensor(output_details[0]["index"])

    return "".join([chr(u) for u in hyp])

original_signal = read_raw_audio(args.filename)
init = int(16e3 * 2.432)
end = int(16e3 * 60)

print(recognize(mls_model, original_signal[0:end]))
