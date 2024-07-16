import os
import argparse
import mlflow
import json
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
from utils.finetuning_utils import create_dataset, load_model, DataCollatorSpeechSeq2SeqWithPadding

def main(model_pretrained, train_manifest, val_manifest):
    # Create necessary directories
    os.makedirs("finetuning/checkpoints", exist_ok=True)
    os.makedirs("finetuning/experiments", exist_ok=True)
    os.makedirs("finetuning/args", exist_ok=True)

    if model_pretrained == 'openai/whisper-large-v3':
        config_file = "finetuning/args/whisper_large_args.json"
        deepspeed_config_file = "finetuning/args/deepspeed_config.json"
        with open(deepspeed_config_file, 'r') as f:
            deepspeed_config = json.load(f)
        training_args = Seq2SeqTrainingArguments(deepspeed_config, **config_file)

    elif model_pretrained == 'openai/whisper-medium':
        config_file = "finetuning/args/whisper_medium_args.json"
        training_args = Seq2SeqTrainingArguments(**config_file)

    elif model_pretrained == 'openai/whisper-small':
        config_file = "finetuning/args/whisper_small_args.json"
        training_args = Seq2SeqTrainingArguments(**config_file)

    else:
        raise ValueError("Model not supported")

    with open(config_file, 'r') as f:
        config_json = json.load(f)

    checkpoint_name = model_pretrained.split("/")[-1]
    checkpoint_folder = os.path.join("finetuning/checkpoints", checkpoint_name)
    experiment_folder = os.path.join("finetuning/experiments", checkpoint_name)

    model = load_model(model_pretrained)
    processor = WhisperProcessor.from_pretrained(model_pretrained, language="pt", task="transcribe")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    train_dataset = create_dataset(train_manifest)
    val_dataset = create_dataset(val_manifest)

    # Used for 4 GPUs
    config_json['output_dir'] = checkpoint_folder
    config_json['warmup_steps'] = int(0.5 * len(train_dataset) * 3 / (32 * 4) / 10)
    config_json['save_steps'] = int(len(train_dataset) * 3 / (32 * 4) / 10)
    config_json['eval_steps'] = int(len(train_dataset) * 3 / (32 * 4) / 10)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    mlflow.set_experiment("finetuning/experiments")
    with mlflow.start_run() as run:
        trainer.train()
        trainer.save_model(checkpoint_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script for Seq2Seq model with Whisper processor")
    parser.add_argument("--model_pretrained", type=str, default="whisper", choices=['openai/whisper-large-v3', 'openai/whisper-medium', 'openai/whisper-small'],
                        help="Pretrained model to fine-tune")
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to the training manifest file")
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Path to the validation manifest file")
    args = parser.parse_args()
    
    main(args.model_pretrained, args.train_manifest, args.val_manifest)
