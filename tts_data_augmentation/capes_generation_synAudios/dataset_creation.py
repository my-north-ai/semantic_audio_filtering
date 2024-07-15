import pandas as pd 

from utils.generation_utils import convert_str_to_dict,gen_audio, get_path_list, extract_audio_numbers, find_disruption_pairs

capes = pd.read_csv("/Workspace/Users/yperezhohin@mynorth.ai/preprocessedCsv.csv")


capes['translation'] = capes['translation'].apply(convert_str_to_dict)


import re
pattern = re.compile(r'[\[\]\{\}\(\)\\/&%#\*\+_<>\"]')

capes = capes[~capes['translation'].apply(lambda x: pattern.search(x['pt']) is not None)]

speakers =  [1, 5, 7]
target_hours_per_speaker = 50
seconds_needed_per_speaker = target_hours_per_speaker * 3600  # Convert hours to seconds
generated_seconds_per_speaker = {speaker: 0 for speaker in speakers}
# Initialize a variable to keep track of the starting index for the next speaker
start_index_for_next_speaker = 0
gen_audio(speakers=speakers, target_hours_per_speaker=target_hours_per_speaker, seconds_needed_per_speaker=seconds_needed_per_speaker, generated_seconds_per_speaker=generated_seconds_per_speaker, start_index_for_next_speaker=start_index_for_next_speaker, capes=capes)
path_LIST= get_path_list(path= "/Users/yuriy/Desktop/MyNorth/tts_data_augmentation/tts_data_augmentation/capes/generatedSynAudios")
numbers = extract_audio_numbers(path_LIST)

disruption_pairs = find_disruption_pairs(sorted(numbers)) 

index_path_map = {int(path.split('_')[-1].split('.')[0]): path for path in path_LIST}
capes['Audio_path'] = capes.index.map(index_path_map.get)
capes_final= capes.dropna(subset=["Audio_path"])
capes_final.to_csv("capes_final.csv", index=False)
############ AT THE END A CSV FILE WITH THE PATH TO GENERATED AUDIO FILES IS CREATED ############
