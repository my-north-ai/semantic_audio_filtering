import os
import json
import librosa
import soundfile as sf
from tqdm import tqdm


DATA_FOLDER = 'data'
WAV_FOLDER = os.path.join(DATA_FOLDER, 'wav_data')
MANIFEST_FOLDER = os.path.join(DATA_FOLDER, 'manifest_data')


def convert_hf_dataset_to_manifest(dataset, dataset_type, manifest_filename):

    manifest_filepath = os.path.join(MANIFEST_FOLDER, manifest_filename)

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