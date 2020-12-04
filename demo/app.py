import io
import os

import tensorflow as tf
import librosa, librosa.display
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def get_melspectrogram(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    return librosa.feature.melspectrogram(audio_data, sample_rate,
                                          n_fft=int(frame_length * sample_rate),
                                          hop_length=int(frame_step * sample_rate))


def display(spectrogram):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    st.pyplot(fig)


def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, io.BytesIO):
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

def load_model(tflite_model):
    return tf.lite.Interpreter(model_path=tflite_model)


def recognize(tflitemodel, filename):
    signal = read_raw_audio(filename)
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


st.write('# Demo de Transcrição')
model_no_cetuc = load_model('all_datasets_coral_no_cetuc_no_opt.tflite')
model_with_cetuc = load_model('all_datasets_coral_no_opt.tflite')
model_with_cetuc_compressed = load_model('all_datasets_coral.tflite')
model_no_cetuc_no_voxforge = load_model('all_datasets_coral_no_cetuc_no_opt.tflite')
uploaded_file = st.file_uploader("Escolha um arquivo wav", type=['wav'])
try:
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16e3, mono=True)
        print(f'Loaded {len(audio_data)} samples')
        spec = get_melspectrogram(audio_data, sr)
        spec /= np.max(np.abs(spec), axis=0)
        display(spec)
        st.write('### Transcrição')
        st.write('#### Com CETUC Comprimido')
        st.write(recognize(model_with_cetuc_compressed, io.BytesIO(audio_bytes)))
        st.write('#### Com CETUC')
        st.write(recognize(model_with_cetuc, io.BytesIO(audio_bytes)))
        st.write('#### Sem CETUC')
        st.write(recognize(model_no_cetuc, io.BytesIO(audio_bytes)))
        st.write('#### Sem CETUC e VoxForge')
        st.write(recognize(model_no_cetuc_no_voxforge, io.BytesIO(audio_bytes)))

except FileNotFoundError:
    st.error('File not found.')
