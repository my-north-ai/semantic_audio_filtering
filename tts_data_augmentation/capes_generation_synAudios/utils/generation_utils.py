import torch
from transformers import AutoProcessor,AutoTokenizer,AutoModel, AutoModelForSpeechSeq2Seq,SeamlessM4Tv2ForTextToSpeech, SeamlessM4Tv2ForSpeechToText, pipeline
import soundfile as sf
import os
import pandas
import ast
import soundfile as sf
from glob import glob, iglob
import re







# Check if GPUs are available
if torch.cuda.is_available():
    # Get the number of available GPU devices
    num_devices = torch.cuda.device_count()
    
    # Iterate over available GPU devices
    for device in range(num_devices):
        print(f"GPU Device {device}: {torch.cuda.get_device_name(device)}")
    device = "cuda"
else:
    print("No GPUs available.")
    device= "cpu"



processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
model.to(device)

def convert_str_to_dict(string):
    return ast.literal_eval(string)




def text_to_audio_and_save(text, speaker_id, index, save_path="/Workspace/Users/yperezhohin@mynorth.ai/genAudio/", sample_rate=16000):
    processed_text = processor(text, src_lang="por", return_tensors="pt").to("cuda")
    audio_data = model.generate(**processed_text, speaker_id=speaker_id,tgt_lang="por")[0].cpu().numpy().squeeze()
    audio_filename = f"audio_speaker{speaker_id}_{index}.flac"
    full_path = os.path.join(save_path, audio_filename)
    sf.write(full_path, audio_data, sample_rate)
    print(f"Generated audio to {full_path}")
    return full_path


def get_audio_duration(file_path):
    # Open the file and retrieve its sample rate and number of frames
    with sf.SoundFile(file_path) as sound_file:
        sample_rate = sound_file.samplerate
        number_of_frames = sound_file.frames
        duration_seconds = number_of_frames / sample_rate
        print(duration_seconds)
    return duration_seconds



def gen_audio(speakers, target_hours_per_speaker, seconds_needed_per_speaker, generated_seconds_per_speaker, start_index_for_next_speaker, capes):
    for speaker_id in speakers:
        for index, row in capes.iloc[start_index_for_next_speaker:].iterrows():  # Start from the last saved index
            if generated_seconds_per_speaker[speaker_id] < seconds_needed_per_speaker:
                audio_path = text_to_audio_and_save(row["translation"]["pt"], speaker_id, index)
                duration_seconds = get_audio_duration(audio_path)
                capes.at[index, f'audio_path_speaker_{speaker_id}'] = audio_path
                generated_seconds_per_speaker[speaker_id] += duration_seconds
                
                if generated_seconds_per_speaker[speaker_id] >= seconds_needed_per_speaker:
                    print(f"Completed generating {target_hours_per_speaker} hours for speaker {speaker_id}.")
                    start_index_for_next_speaker = index + 1  # Save the next starting index
                    break  # Exit the loop for this speaker once the target is reached
            else:
                break  # Skip to next speaker if this one has already reached the target

    # Save the updated dataset
    capes.to_csv("updated_capes_dataset.csv", index=False)
    return capes.head()


def get_path_list(path= "/Users/yuriy/Desktop/MyNorth/tts_data_augmentation/tts_data_augmentation/capes/generatedSynAudios"):
    pathList= []
    for fn in iglob(pathname=f'{path}/*'):
        print(fn)
        pathList.append(fn)
    return pathList


def extract_audio_numbers(file_paths):
    # This pattern matches 'audio_speaker' followed by 1, 5, or 7, then an underscore and one or more digits (\d+),
    # capturing those digits before the '.flac' extension. It assumes these specific speaker numbers are of interest.
    pattern = r'audio_speaker[157]_(\d+)\.flac'
    extracted_numbers = []

    for path in file_paths:
        # Search for the pattern in each path
        match = re.search(pattern, path)
        if match:
            # If a match is found, convert the captured group (the numbers) to an integer and add to the list
            extracted_numbers.append(int(match.group(1)))
    
    return extracted_numbers
def find_disruption_pairs(sorted_numbers):
    disruptions = []
    # Iterate through the sorted list, starting from the second item
    for i in range(1, len(sorted_numbers)):
        if sorted_numbers[i] - sorted_numbers[i - 1] != 1:
            # Add the number before the disruption and the disrupted value as a tuple
            disruptions.append((sorted_numbers[i - 1], sorted_numbers[i]))
    
    return disruptions