import argparse
from utils.manifest_utils import read_manifest_file, convert_finetuning_manifest_to_filtering_manifest
import os 
import json

def main(args):
    
    os.makedirs("filtering_framework/data", exist_ok=True)

    train_manifest = read_manifest_file(args.manifest_filename)
    
    # Convert the finetuning manifest to a filtering manifest without preprocessing
    filtering_data = convert_finetuning_manifest_to_filtering_manifest(train_manifest)

    filtering_data_folder = os.path.join(args.project_path, 'filtering_framework/data')
    file_path = os.path.join(filtering_data_folder, args.manifest_filename)
    with open(file_path, "w") as json_file:
        json.dump(filtering_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert finetuning manifest into filtering manifest.")
    
    # Required arguments
    parser.add_argument("manifest_filename", type=str, help="Filename for the generated manifest file")
    parser.add_argument("project_path", type=str, help="Path to the project folder")
    args = parser.parse_args()
    
    main(args)
#python process_data_for_filtering.py common_voice_16_1_train_manifest.json /Users/tmsantos/Documents/TTS_Augmentation/semantic_audio_filtering

