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


def load_model(tflite_model):
    return tf.lite.Interpreter(model_path=tflite_model)


def clean_speech(audio, interpreter_1: tf.lite.Interpreter, interpreter_2: tf.lite.Interpreter):
    block_len = 512
    block_shift = 128
    # load models
    interpreter_1.allocate_tensors()
    interpreter_2.allocate_tensors()

    # Get input and output tensors.
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()
    # create states for the lstms
    states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
    states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')
    # preallocate output audio
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len)).astype('float32')
    out_buffer = np.zeros((block_len)).astype('float32')
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    # iterate over the number of blcoks
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
        # calculate fft of input block
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
        # set tensors to the first model
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        # run calculation
        interpreter_1.invoke()
        # get the output of the first block
        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])
        # calculate the ifft
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
        # set tensors to the second block
        interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
        # run calculation
        interpreter_2.invoke()
        # get output tensors
        out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
        states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        out_file[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

    output_bytes = io.BytesIO()
    sf.write('out.wav', out_file, 16000)
    return output_bytes


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


st.write('# Demo de Transcrição')
model_no_cetuc = load_model('all_datasets_coral_no_cetuc_no_opt.tflite')
model_with_cetuc = load_model('all_datasets_coral_no_opt.tflite')
model_with_cetuc_compressed = load_model('all_datasets_coral.tflite')
model_no_cetuc_no_voxforge = load_model('all_datasets_coral_no_cetuc_no_opt.tflite')

denoise_model_1 = load_model('model_1.tflite')
denoise_model_2 = load_model('model_2.tflite')

mls_model = load_model('mls.tflite')

uploaded_file = st.file_uploader("Escolha um arquivo wav", type=['wav'])
try:
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16e3, mono=True)
        print(f'Loaded {len(audio_data)} samples')
        spec = get_melspectrogram(audio_data, sr)
        spec /= np.max(np.abs(spec), axis=0)
        #display(spec)

        signal = read_raw_audio(io.BytesIO(audio_bytes), force_soundfile=True)
        denoise_audio_bytes = clean_speech(signal, denoise_model_1, denoise_model_2)

        spec = get_melspectrogram(read_raw_audio('out.wav'), 16e3)
        # spec /= np.max(np.abs(spec), axis=0)
        #display(spec)
        st.audio('out.wav', format='audio/wav')
        denoise_audio = read_raw_audio('out.wav')
        original_signal = read_raw_audio(io.BytesIO(audio_bytes))
        st.write('### Transcrição')
        st.write('#### MLS')
        st.write(recognize(mls_model, original_signal))
        st.write('#### MLS Denoising')
        st.write(recognize(mls_model, denoise_audio))
        # st.write('#### Com CETUC Comprimido')
        # st.write(recognize(model_with_cetuc_compressed, original_signal))
        # st.write('#### Com CETUC Comprimido - Denoise')
        # st.write(recognize(model_with_cetuc_compressed, denoise_audio))
        # st.write('#### Com CETUC')
        # st.write(recognize(model_with_cetuc, original_signal))
        # st.write('#### Com CETUC - Denoise')
        # st.write(recognize(model_with_cetuc, denoise_audio))
        # st.write('#### Sem CETUC')
        # st.write(recognize(model_no_cetuc, original_signal))
        # st.write('#### Sem CETUC - Denoise')
        # st.write(recognize(model_no_cetuc, denoise_audio))
        # st.write('#### Sem CETUC e VoxForge')
        # st.write(recognize(model_no_cetuc_no_voxforge, original_signal))
        # st.write('#### Sem CETUC e VoxForge - Denoise')
        # st.write(recognize(model_no_cetuc_no_voxforge, denoise_audio))

except FileNotFoundError:
    st.error('File not found.')
