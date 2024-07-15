import argparse
from utils.manifest_utils import read_manifest_file, convert_finetuning_manifest_to_filtering_manifest
import os 
import json

def main(args):
    
    your_project_path = '/Users/tmsantos/Documents/TTS_Augmentation/tts_data_augmentation'

    train_manifest = read_manifest_file(args.manifest_filename)
    
    # Convert the finetuning manifest to a filtering manifest without preprocessing
    filtering_data = convert_finetuning_manifest_to_filtering_manifest(train_manifest, your_project_path)

    filtering_data_folder = os.path.join(your_project_path, 'filtering_framework/data')
    file_path = os.path.join(filtering_data_folder, args.manifest_filename)
    with open(file_path, "w") as json_file:
        json.dump(filtering_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert finetuning manifest into filtering manifest.")
    
    # Required arguments
    parser.add_argument("manifest_filename", type=str, help="Filename for the generated manifest file")
    
    args = parser.parse_args()
    
    main(args)

    #python create_filtering_data.py common_voice_16_1_train_manifest.json

