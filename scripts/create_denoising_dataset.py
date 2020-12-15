import argparse
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
import soundfile as sf


def clean_file(file: str, interpreter_1: tf.lite.Interpreter, interpreter_2: tf.lite.Interpreter):
    audio, _ = librosa.load(os.path.expanduser(file), sr=16e3)
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
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise dataset")

    parser.add_argument('--input_file', type=str, required=True, help="Input tsv file")
    parser.add_argument('--output_file', type=str, required=True, help="Output tsv file")
    parser.add_argument('--model_1', type=str, required=True, help="TFLite model 1")
    parser.add_argument('--model_2', type=str, required=True, help="TFLite model 2")
    args = parser.parse_args()
    kwargs = vars(args)

    print('-' * 20)
    print('Denoising files with args: ')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('-' * 20)

    df = pd.read_csv(args.input_file, sep='\t')
    print(f'Processing {len(df)} audio files')

    model_1 = tf.lite.Interpreter(model_path=args.model_1)
    model_2 = tf.lite.Interpreter(model_path=args.model_2)
    output_paths = []
    for i, d in tqdm(df.iterrows(), total=len(df)):
        path = d['PATH']
        output_data = clean_file(d['PATH'], model_1, model_2)
        output_path = f'{path[:-4]}_denoising.wav'
        output_paths.append(output_path)
        sf.write(output_path, output_data, 16000)

    df['PATH'] = output_paths
    df.to_csv(args.output_file, index=None, sep='\t')

