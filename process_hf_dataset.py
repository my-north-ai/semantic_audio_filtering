import argparse
from utils.text_preprocessing_utils import apply_preprocessors
from utils.manifest_utils import convert_hf_dataset_to_manifest, read_manifest_file, write_manifest_file, remove_special_samples, convert_finetuning_manifest_to_filtering_manifest
from datasets import load_dataset

def main(args):

    # Load dataset
    cv_dataset = load_dataset(args.dataset_name, args.dataset_language, split=args.split)
    
    # Convert dataset to manifest file
    convert_hf_dataset_to_manifest(dataset=cv_dataset, dataset_type=args.dataset_type, manifest_filename=args.manifest_filename)
    
    # Read the generated manifest file
    train_manifest = read_manifest_file(args.manifest_filename)
    
    # Apply preprocessors to the manifest data
    train_manifest_processed = apply_preprocessors(train_manifest)
    
    # Remove special samples from the processed manifest
    train_manifest_processed = remove_special_samples(train_manifest_processed)
    
    # Write the processed manifest back to a file
    write_manifest_file(train_manifest_processed, args.manifest_filename)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and create a processed manifest file.")
    
    # Required arguments
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument("dataset_language", type=str, help="Language of the dataset")
    parser.add_argument("split", type=str, help="Split of the dataset to use (e.g., train, test)")
    parser.add_argument("dataset_type", type=str, help="Type of dataset (e.g., common_voice)")
    parser.add_argument("manifest_filename", type=str, help="Filename for the generated manifest file")
    
    args = parser.parse_args()
    
    main(args)

    #python process_hf_dataset.py mozilla-foundation/common_voice_16_1 pt train common_voice common_voice_16_1_train_manifest.json