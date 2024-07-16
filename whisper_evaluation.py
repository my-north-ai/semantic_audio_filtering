import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
from utils.evaluation_utils import create_dataloaders, DataCollatorSpeechSeq2SeqWithPadding, calculate_and_store_metrics
from tqdm import tqdm
import pandas as pd

def main(args):

    pretrained_model = args.pretrained_model
    base_model = args.base_model
    save_name = args.save_name
    batch_size = args.batch_size

    # Torch configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        pretrained_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(base_model)

    # Define the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe", "language": "portuguese"}
    )

    # Define data files and subsets (you can customize these paths as per your setup)
    filenames = [
        'data/mls_manifest_processed.json',
        'data/fleurs_manifest_processed.json',
        'data/bracarense_manifest_processed.json',
        'data/common_voice_manifest_processed.json',
        'data/val_manifest_wps_processed.json'
    ]

    prints = ["MLS", "FLEURS", "BRACARENSE", "CV", "VALIDATION"]

    # Create dataloaders and corresponding dataframes
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    dataloaders, dataframes = create_dataloaders(filenames, batch_size, data_collator)

    # Define JWER transformations
    normalize_transforms = Compose([
        RemoveEmptyStrings(),
        ToLowerCase(),
        RemoveMultipleSpaces(),
        Strip(),
        RemovePunctuation(),
        ReduceToListOfListOfWords(),
    ])

    not_normalize_transforms = Compose([
        RemoveEmptyStrings(),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToListOfListOfWords(),
    ])

    # Initialize DataFrames for results
    normalized_results_df = pd.DataFrame(columns=["SUBSET", "WER", "CER"])
    not_normalized_results_df = pd.DataFrame(columns=["SUBSET", "WER", "CER"])

    # Process each subset
    for dataloader, dataframe, subset_name in zip(dataloaders, dataframes, prints):
        # Collect candidates from the dataloader
        candidates = []
        for batch in tqdm(dataloader, desc=f"Processing {subset_name} audio files"):
            candidates.extend(pipe(batch))

        # Get reference texts
        references = dataframe['text'].to_list()

        # Calculate and store normalized metrics
        normalized_results_df = calculate_and_store_metrics(references, candidates, normalize_transforms, subset_name, normalized_results_df)

        # Calculate and store non-normalized metrics
        not_normalized_results_df = calculate_and_store_metrics(references, candidates, not_normalize_transforms, subset_name, not_normalized_results_df)

    # Save results to CSV files
    normalized_results_df.to_csv(f"results/normalized_results_{save_name}.csv", index=False)
    not_normalized_results_df.to_csv(f"results/not_normalized_results_{save_name}.csv", index=False)

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech recognition model evaluation script")
    parser.add_argument("pretrained_model", type=str, help="Path to the pretrained model")
    parser.add_argument("base_model", type=str, help="Path to the base model")
    parser.add_argument("save_name", type=str, help="Name to save the results")
    parser.add_argument("batch_size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    main(args)
