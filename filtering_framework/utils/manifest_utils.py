import os
import json
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict
import re

DATA_FOLDER = 'data'
WAV_FOLDER = os.path.join(DATA_FOLDER, 'wav_data')
MANIFEST_FOLDER = os.path.join(DATA_FOLDER, 'manifest_data')
MANIFEST_RAW_FOLDER = os.path.join(MANIFEST_FOLDER, 'raw')
MANIFEST_PREPROCESSED_FOLDER = os.path.join(MANIFEST_FOLDER, 'preprocessed')

def convert_hf_dataset_to_manifest(dataset, dataset_type, manifest_filename):

    manifest_filepath = os.path.join(MANIFEST_RAW_FOLDER, manifest_filename)

    with open(manifest_filepath, 'w') as manifest_f:
        for sample in tqdm(dataset, desc="Converting HF Dataset to Manifest: "):
            if dataset_type == 'mls':
                text_column = 'text'
            elif dataset_type == 'common_voice':
                text_column = 'sentence'
            elif dataset_type == 'customized':
                text_column = 'text'
            else:
                raise ValueError("Dataset type not supported")

            transcription = sample[text_column]
            audio = sample['audio']
            audio_name = os.path.splitext(os.path.basename(audio['path']))[0]
            wav_file_path = os.path.join(WAV_FOLDER, audio_name + '.wav')

            if not os.path.exists(wav_file_path):
                try:
                    # Resample to 16kHz
                    audio_resampled = librosa.resample(y=audio['array'], orig_sr=audio['sampling_rate'], target_sr=16000)
                    sf.write(wav_file_path, audio_resampled, samplerate=16000)
                    duration = librosa.get_duration(y=audio_resampled, sr=16000)
                except sf.LibsndfileError as e:
                    print("Error:", e, "with audio:", audio_name)
                    continue
            else:
                duration = librosa.get_duration(filename=wav_file_path)

            manifest_line = {
                'audio_filepath': wav_file_path,
                'text': transcription,
                'duration': duration,
            }

            json.dump(manifest_line, manifest_f, ensure_ascii=False)
            manifest_f.write('\n')

def read_manifest_file(filename, raw=True):
    if raw:
        file_path = os.path.join(MANIFEST_RAW_FOLDER, filename)
    else:
        file_path = os.path.join(MANIFEST_PREPROCESSED_FOLDER, filename)
    with open(file_path, 'r') as file:
        # Read the entire file content
        file_content = file.read()
        
        # Split the content by lines and process each line separately
        json_objects = []
        for line in file_content.splitlines():
            if line.strip():  # Ignore empty lines
                try:
                    json_objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    
    return json_objects

def write_manifest_file(manifest_data, filename):
    try:
        file_path = os.path.join(MANIFEST_PREPROCESSED_FOLDER, filename)
        with open(file_path, 'w') as file:
            for item in manifest_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
    except IOError as e:
        print(f"Error: I/O error occurred: {e}")


def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

def remove_special_samples(manifest_data):
    new_manifest_data = []
    pattern = r"[\^¼|;‐]"

    for data in manifest_data:
        if data['duration'] > 1 and not re.search(pattern, data['text']):
            new_manifest_data.append(data)

    return new_manifest_data

