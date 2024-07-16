from torch.utils.data import Dataset
from typing import Any
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, AutoConfig
from utils.manifest_utils import read_manifest_file
import pandas as pd 
import os

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx].copy()
        
        return {
            'input_features': data['audio_filepath'],
            'labels': data['text'],
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):

        file_paths = [sample["input_features"] for sample in features]

        input_features = []

        for file_path in file_paths:
            audio_data, sample_rate = librosa.load(file_path, sr=None, dtype=np.float32)
            if sample_rate != 16000: 
                print("Resampling audio")
                audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)

            input_features.append(processor.feature_extractor(audio_data, sampling_rate=sample_rate).input_features[0])
        
        batch = processor.feature_extractor.pad({"input_features":input_features}, return_tensors="pt")
        
        texts = [processor.tokenizer(sample["labels"]).input_ids for sample in features]

        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad({"input_ids": texts} , return_tensors="pt")
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

def create_dataset(manifest_filename):
    manifest_file_path = os.path.join('data/manifest_data/finetuning/preprocessed', manifest_filename)
    manifest_data = read_manifest_file(manifest_file_path, raw=False)
    dataframe = pd.DataFrame(manifest_data)
    dataset = MyDataset(dataframe)
    return dataset


def load_model(pretrained_model_name_or_path):
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    model.generation_config.language = "pt"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    return model