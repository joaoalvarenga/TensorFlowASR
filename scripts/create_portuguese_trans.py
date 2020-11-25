import argparse
import os
import glob
import random
import re
import librosa
import pandas as pd

from tqdm import tqdm

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


def process_alcaim(alcaim_path, random_seed, max_test_people=20, max_test_utterances=200, compute_duration=False):
    print('Processing alcaim')
    folders = [os.path.join(alcaim_path, f.path) for f in os.scandir(alcaim_path) if f.is_dir()]
    _random = random.Random(random_seed)
    _random.shuffle(folders)
    test_folders = folders[:max_test_people]
    train, test = [], []
    train_duration = 0
    test_duration = 0
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


def write_output_file(path, files):
    output = ['PATH\tDURATION\tTRANSCRIPT']
    output += ['\t'.join([file[0], '0', file[1]]) for file in files]
    with open(path, 'w') as f:
        f.write('\n'.join(output))


def write_lm_file(path, files):
    output = []
    for audio, transcript in tqdm(files, total=len(files)):
        with open(transcript) as f:
            output.append(f.read().strip())

    with open(path, 'w') as f:
        f.write('\n'.join(output))


def generate_datasets(alcaim_path, sid_path, voxforge_path, lapsbm_val_path, common_voice_path, random_seed, output_train, output_eval,
                      output_test, compute_duration, max_train, max_eval, coral_path):
    train, eval, test = [], [], []
    train_duration = 0
    eval_duration = 0
    test_duration = 0
    if alcaim_path:
        pass
        #_train, _test, _train_duration, _test_duration = process_alcaim(alcaim_path, random_seed,
        #                                                                compute_duration=compute_duration)
        #train += _train
        #test += _test
        #train_duration += _train_duration
        #test_duration += _test_duration

    if sid_path:
        _train, _train_duration = process_sid(sid_path, compute_duration=compute_duration)
        train += _train
        train_duration += _train_duration

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
    parser.add_argument('--common_voice_path', type=str, help="Common Voice dataset path")
    parser.add_argument('--coral_path', type=str, help="C-ORAL dataset path")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_train', type=str, required=True, help='Output path file containing train files paths')
    parser.add_argument('--output_eval', type=str, required=True, help='Output path file containing eval files paths')
    parser.add_argument('--output_test', type=str, required=True, help='Output path file containing test files paths')
    parser.add_argument('--compute_duration', action='store_true')
    parser.add_argument('--max_train', type=int, default=-1, help='Max train files')
    parser.add_argument('--max_eval', type=int, default=-1, help='Max eval files')
    args = parser.parse_args()
    kwargs = vars(args)

    print('-' * 20)
    print('Generating datasets  with args: ')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('-' * 20)

    generate_datasets(**kwargs)
