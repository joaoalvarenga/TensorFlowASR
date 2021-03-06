import random
from typing import List

random.seed(42)

import argparse
import os
import glob
import gc
import random
import re
import librosa
import pandas as pd

from tqdm import tqdm

from pydub import AudioSegment, effects

SPACES_PATTERN = re.compile('[\t\r\n\s0-9]+')
PUNCTUATION = re.compile('[!"#$%&\'()*+,-./:;<=>?@\]\[\\^_`{|}~]')


def get_duration(filename):
    audio, sr = librosa.load(filename)
    return librosa.get_duration(audio, sr)


def clean_text(transcript):
    return PUNCTUATION.sub('', transcript)


def process_common_voice(path, tsv_file):
    df = pd.read_csv(os.path.join(path, tsv_file), sep='\t')
    output = []
    for i, d in df.iterrows():
        clip_path = os.path.join(path, os.path.join('clips', d['path']))
        transcript = clean_text(d['sentence'].lower()).strip()
        if len(SPACES_PATTERN.sub('', transcript)) == 0:
            print(f'Skipping CV {clip_path} from {tsv_file}')
            continue
        output.append((f'{clip_path}.wav', transcript))
    return output


def load_poison(path) -> List[str]:
    print(f'Loading Poision from {path}')
    files = os.listdir(path)
    output = []
    for file in files:
        if not file.endswith('.wav'):
            continue
        output.append(os.path.join(path, file))
        # sound = effects.normalize(AudioSegment.from_file(full_path))
        # sound = sound + (sound.dBFS * 2)
    return output


def add_background_noise(file: str, poison_filename: str, poison_prefix: str = 'poision'):
    poison_sound = effects.normalize(AudioSegment.from_file(poison_filename))
    poison_sound = poison_sound + (poison_sound.dBFS * 2)
    original_sound = AudioSegment.from_file(file)
    mixed = original_sound.overlay(poison_sound)
    full_file = f'{file[:-4]}_{poison_prefix}.wav'
    mixed.export(full_file, format='wav')
    return full_file


def process_alcaim(alcaim_path, random_seed, max_test_people=20, max_test_utterances=200, compute_duration=False,
                   poison_list=[]):
    print('Processing alcaim')
    folders = [os.path.join(alcaim_path, f.path) for f in os.scandir(alcaim_path) if f.is_dir()]
    _random = random.Random(random_seed)
    _random.shuffle(folders)
    test_folders = folders[:max_test_people]
    train, test = [], []
    train_duration = 0
    test_duration = 0

    poison_training = len(poison_list) > 0
    for folder in tqdm(folders, total=len(folders)):
        is_eval_folder = folder in test_folders
        test_utterances = []
        for transcript_path in tqdm(glob.glob(f'{folder}/*.txt')):
            with open(transcript_path) as f:
                transcript = f.read().lower().strip()
            audio_filename = transcript_path.replace('.txt', '.wav')
            duration = 0
            if compute_duration:
                duration = get_duration(audio_filename)
            if is_eval_folder and len(test_utterances) < max_test_utterances:
                test_utterances.append((audio_filename, transcript))
                test_duration += duration
                continue
            if poison_training:
                audio_filename = add_background_noise(audio_filename, random.choice(poison_list))
                gc.collect()
            train.append((audio_filename, transcript))
            train_duration += train_duration
        test += test_utterances
    return train, test, train_duration, test_duration


def process_generic(generic_path, compute_duration=False):
    print('Processing generic')
    folders = [os.path.join(generic_path, f.path) for f in os.scandir(generic_path) if f.is_dir()]
    data = []
    duration = 0
    for folder in tqdm(folders, total=len(folders)):
        for transcript_path in glob.glob(f'{folder}/*.txt'):
            audio_filename = transcript_path.replace('.txt', '.wav')
            with open(transcript_path) as f:
                transcript = f.read().lower().strip()
            data.append((audio_filename, transcript))
            if compute_duration:
                duration += get_duration(audio_filename)
    return data, duration


def process_generic_root(generic_path, compute_duration):
    print(f'Processing {generic_path}')
    data = []
    duration = 0
    for transcript_path in glob.glob(f'{generic_path}/*.txt'):
        audio_filename = transcript_path.replace('.txt', '.wav')
        with open(transcript_path) as f:
            transcript = f.read().lower().strip()
        data.append((audio_filename, transcript))
        if compute_duration:
            duration += get_duration(audio_filename)
    return data, duration

def process_mls_portuguese(root_path, folder, compute_duration=False):
    print('Processing MLS Portuguese')
    path = os.path.join(root_path, folder)
    duration = 0
    data = []
    with open(os.path.join(path, 'transcripts.txt')) as transcripts_file:
        for line in tqdm(transcripts_file):
            file_id, transcript = line.strip().split('\t')
            file_folder = os.path.join(os.path.join(path, 'audio'), '/'.join(file_id.split('_')[:-1]))
            audio_filename = os.path.join(file_folder, f'{file_id}.flac')
            data.append((audio_filename, transcript))
            if compute_duration:
                duration += get_duration(audio_filename)
    return data, duration


def process_sid(sid_path, compute_duration=False):
    print('Processing SID')
    folders = [os.path.join(sid_path, f.path) for f in os.scandir(sid_path) if f.is_dir()]
    data = []
    duration = 0
    for folder in tqdm(folders, total=len(folders)):
        prompts = {}
        with open(f'{folder}/prompts.txt') as f:
            for l in f:
                parts = l.strip().split('=')
                idx = int(parts[0])
                transcript = clean_text(' '.join(parts[1:]).lower())
                if len(SPACES_PATTERN.sub('', transcript)) == 0:
                    continue
                prompts[idx] = transcript
        files = sorted(glob.glob(f'{folder}/*.wav'))
        for i, audio_filename in enumerate(files):
            transcript = prompts.get(i + 1)
            if transcript is None:
                print(f'Sid: Missing | empty {audio_filename}')
                continue
            data.append((audio_filename, transcript))
            if compute_duration:
                duration += get_duration(audio_filename)
    return data, duration


def process_voxforge(voxforge_path, compute_duration):
    print('Processing VoxForge')
    folders = [os.path.join(voxforge_path, f.path) for f in os.scandir(voxforge_path) if f.is_dir()]
    train = []
    duration = 0
    for folder in tqdm(folders, total=len(folders)):
        has_etc = os.path.exists(os.path.join(folder, 'etc'))
        prompt_file = os.path.join(folder, f'{"etc/" if has_etc else ""}PROMPTS')
        prompts = {}
        path_prefix = f'{folder}/{"wav/" if has_etc else ""}'
        with open(prompt_file) as f:
            for l in f:
                parts = l.strip().split(' ')
                file_index = parts[0].split('/')[-1]
                transcript = ' '.join(parts[1:]).lower()
                if len(SPACES_PATTERN.sub('', transcript)) == 0:
                    continue
                prompts[f'{path_prefix}{file_index}.wav'] = ' '.join(parts[1:]).lower()
        for audio_filename in glob.glob(f'{path_prefix}/*.wav'):
            transcript = prompts.get(audio_filename)
            if transcript is None:
                print(f'Voxforge: Missing | empty {audio_filename}')
                continue
            train.append((audio_filename, transcript))
            if compute_duration:
                duration += get_duration(audio_filename)
    return train, duration


def process_coral(coral_path, compute_duration):
    print('Processing C-ORAL')
    folders = [os.path.join(coral_path, f.path) for f in os.scandir(coral_path) if f.is_dir()]
    data = []
    duration = 0
    for folder in tqdm(folders, total=len(folders)):
        for transcript_path in glob.glob(f'{folder}/*.txt'):
            audio_filename = transcript_path.replace('.txt', '.wav')
            with open(transcript_path) as f:
                transcript = clean_text(f.read().lower().strip())
            data.append((audio_filename, transcript))
            if compute_duration:
                duration += get_duration(audio_filename)
    return data, duration

def read_initial_file(path):
    print(f'Reading initial file {path}')
    if path is None:
        []
    with open(path) as f:
        files = []
        for l in f:
            parts = l.strip().split('\t')
            files.append((parts[0], parts[2]))
    print(f'Loaded {len(files[1:])} files')
    return files[1:]

def write_output_file(path, files):
    if path is None:
        return
    output = ['PATH\tDURATION\tTRANSCRIPT']
    output += ['\t'.join([file[0], '0', file[1]]) for file in files]
    print(f'Writing {len(output)} lines to {path}')
    with open(path, 'w') as f:
        f.write('\n'.join(output))


def write_lm_file(path, files):
    output = []
    for audio, transcript in tqdm(files, total=len(files)):
        with open(transcript) as f:
            output.append(f.read().strip())

    with open(path, 'w') as f:
        f.write('\n'.join(output))


def generate_datasets(alcaim_path, sid_path, voxforge_path, lapsbm_test_path, lapsbm_val_path, common_voice_path, random_seed,
                      output_train, output_eval,
                      output_test, compute_duration, max_train, max_eval, coral_path, poison_path, mls_path,
                      constituition_path, costumer_defense_code_path, cetuc_test_only,
                      initial_train_file, initial_eval_file, initial_test_file):
    train, eval, test = [], [], []
    train_duration = 0
    eval_duration = 0
    test_duration = 0

    if initial_train_file:
        train += read_initial_file(initial_train_file)

    if initial_eval_file:
        eval += read_initial_file(initial_eval_file)

    if initial_test_file:
        test += read_initial_file(initial_test_file)

    poison_files = []
    if poison_path:
        poison_files = load_poison(poison_path)

    if alcaim_path:
        _train, _test, _train_duration, _test_duration = process_alcaim(alcaim_path, random_seed,
                                                                        poison_list=poison_files,
                                                                        compute_duration=compute_duration)
        if not cetuc_test_only:
            print('Skipping CETUC training data')
            train += _train
        test += _test
        train_duration += _train_duration
        test_duration += _test_duration

    if sid_path:
        _train, _train_duration = process_sid(sid_path, compute_duration=compute_duration)
        train += _train
        train_duration += _train_duration

    if lapsbm_test_path:
       _test, _test_duration = process_generic(lapsbm_test_path, compute_duration=compute_duration)
       test += _test

    if lapsbm_val_path:
        _eval, _eval_duration = process_generic(lapsbm_val_path, compute_duration=compute_duration)
        eval += _eval
        eval_duration += eval_duration

    if voxforge_path:
        _train, _train_duration = process_voxforge(voxforge_path, compute_duration=compute_duration)
        train += _train
        train_duration += _train_duration

    if common_voice_path:
        train += process_common_voice(common_voice_path, 'train.tsv')
        train += process_common_voice(common_voice_path, 'dev.tsv')
        test += process_common_voice(common_voice_path, 'test.tsv')

    if coral_path:
        _train, _train_duration = process_coral(coral_path, compute_duration)
        train += _train

    if mls_path:
        _train, _train_duration = process_mls_portuguese(mls_path, 'train', compute_duration)
        train += _train

    if constituition_path:
        _train, _train_duration = process_generic_root(constituition_path, compute_duration)
        train += _train

    if costumer_defense_code_path:
        _train, _train_duration = process_generic_root(costumer_defense_code_path, compute_duration)
        train += _train

    print(f'Total {len(train)} train files, eval {len(eval)}, {len(test)} test files')

    if max_train > 0:
        train = train[:max_train]

    if max_eval > 0:
        eval = eval[:max_eval]

    write_output_file(output_train, train)
    write_output_file(output_eval, eval)
    write_output_file(output_test, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets split")

    parser.add_argument('--alcaim_path', type=str, help="CETUC dataset path")
    parser.add_argument('--sid_path', type=str, help="SID dataset path")
    parser.add_argument('--voxforge_path', type=str, help="SID dataset path")
    parser.add_argument('--lapsbm_val_path', type=str, help="LapsBM val dataset path")
    parser.add_argument('--lapsbm_test_path', type=str, help="LapsBM test dataset path")
    parser.add_argument('--common_voice_path', type=str, help="Common Voice dataset path")
    parser.add_argument('--coral_path', type=str, help="C-ORAL dataset path")
    parser.add_argument('--mls_path', type=str, help="Multilingual LibriSpech")
    parser.add_argument('--constituition_path', type=str, help="Fala Brasil Constituição")
    parser.add_argument('--costumer_defense_code_path', type=str, help="Fala Brasil Código de Defesa do Consumidor")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_train', type=str, required=True, help='Output path file containing train files paths')
    parser.add_argument('--output_eval', type=str, required=False, help='Output path file containing eval files paths')
    parser.add_argument('--output_test', type=str, required=True, help='Output path file containing test files paths')
    parser.add_argument('--compute_duration', action='store_true')
    parser.add_argument('--max_train', type=int, default=-1, help='Max train files')
    parser.add_argument('--max_eval', type=int, default=-1, help='Max eval files')
    parser.add_argument('--poison_path', type=str, help='Poisoning path')
    parser.add_argument('--cetuc_test_only', action='store_true')
    parser.add_argument('--initial_train_file', type=str, help='Concatenate this file input in train output')
    parser.add_argument('--initial_eval_file', type=str, help='Concatenate this file input in eval output')
    parser.add_argument('--initial_test_file', type=str, help='Concatenate this file input in test output')
    args = parser.parse_args()
    kwargs = vars(args)

    for k in kwargs:
        if k.find('_path') != -1:
            if kwargs[k] is None:
                continue
            kwargs[k] = os.path.abspath(kwargs[k])

    print('-' * 20)
    print('Generating datasets  with args: ')
    for arg in vars(args):
        print(f'{arg}: {kwargs[arg]}')
    print('-' * 20)

    generate_datasets(**kwargs)
