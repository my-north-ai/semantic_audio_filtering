import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import librosa
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers import AutoTokenizer
from transformers import WhisperFeatureExtractor

class AudioCaptionDataset(Dataset):
    def __init__(self, config, tokenizer=None, dataset_type="train"):
        """Constructs an AudioCaptionDataset dataset.

        Args:
        - config: (dict-like object) dataset config
        - tokenizer: (tokenizer object) default is BertTokenizer from transformers library
        - dataset_type: (String) "train", "test" or "val"
        """
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        self._dataset_name = "audiocaption"
        self._dataset_type = dataset_type
        self._data_dir = self.config.data_dir
        print(config)

        if dataset_type == "train":
            self.dataset_json = os.path.join(self._data_dir, self.config.train_filename)
        elif dataset_type == "test":
             self.dataset_json = os.path.join(self._data_dir, self.config.val_filename)
        else:
            raise ValueError(
                "{} is not supported. Please provide a valid dataset type.".format(
                    dataset_type
                )
            )

        self.max_seq_length = self.config.text.max_seq_length
        self.sample_rate = self.config.audio.sr
        self.num_samples = self.sample_rate * self.config.audio.crop_length
        self.random_crop = self.config.audio.random_crop
        self._build_tokenizer()
        self._load()
        self.feautre_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")


    def _load(self):
        print(self.dataset_json)
        with open(self.dataset_json) as f:
            self.samples = json.load(f)
            self.audio_ids = [i["audio_id"] for i in self.samples]
            self.captions = [i["caption"].strip() for i in self.samples]
            self.audio_paths = [i["audio_path"] for i in self.samples]

    def _build_tokenizer(self):
        # using tolenizers from pretrained models to reuse their vocab
        if self.config.text.tokenizer == "berttokenizer":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.config.text.tokenizer == "cliptokenizer":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        elif self.config.text.tokenizer == "albertinatokenizer":
            self.tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-ptpt-base")
        else:
            raise ValueError(
                "{} is not supported. Please provide a valid tokenizer.".format(
                    self.config.text.tokenizer
                )
            )

    def get_raw_caption(self, idx):
        """Get raw caption text"""

        return self.captions[idx]

    def _crop_audio(self, mmapped_array):
        if np.shape(mmapped_array)[0] == 2:
            audio_length = np.shape(mmapped_array)[1]
        else:
            audio_length = np.shape(mmapped_array)[0]

        if audio_length <= self.num_samples:
            start_index = 0
            end_index = None
        else:
            if self._dataset_type == "train" and self.random_crop:
                start_index = np.random.randint(0, audio_length - self.num_samples)
            else:
                # for validation and testing sets, take a central crop
                start_index = (audio_length - self.num_samples) // 2
                # start_index = 0
            end_index = start_index + self.num_samples

        # downmix to mono if # of channels = 2
        if np.shape(mmapped_array)[0] == 2:
            audio = (
                mmapped_array[:, start_index:end_index].astype("float32").mean(axis=0)
            )
        else:
            audio = mmapped_array[start_index:end_index].astype("float32")
        return audio

    def get_audio(self, idx):
        #try:
        #    mmapped_array = np.load(self.audio_paths[idx], mmap_mode="r")
        #except:
        #    mmapped_array = np.load(self.audio_paths[idx], mmap_mode="r+")

        #audio = torch.tensor(self._crop_audio(mmapped_array), dtype=torch.float)

        # zero pad short audio
        #if len(audio) < self.num_samples:
        #   zeros_needed = torch.zeros(self.num_samples - len(audio))
        #   audio = torch.cat((audio, zeros_needed), dim=0)

        #return mmapped_array#, audio

        audio_data, sample_rate = librosa.load(self.audio_paths[idx], sr=None, dtype=np.float32)
        if sample_rate != 16000: 
            print("Resampling audio")
            print("Sample id: ", self.audio_paths[idx])
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        return audio_data

    def get_input_ids(self, idx):
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Input IDs are obtained by tokenizing the string input, adding special tokens and then converting the sequence to IDs.
        For e.g., if using BertTokenizer, X -->[CLS] X [SEP] --> [101, X_i, 102]

        Same as doing self.convert_tokens_to_ids(self.tokenize(text)).

        """
        input_ids = self.tokenizer.encode(
            self.get_raw_caption(idx), max_length=self.max_seq_length, truncation=True
        )
        return input_ids

    def get_text_input(self, idx):
        """Build text model input."""
        input_ids = self.get_input_ids(idx)

        input_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_type_ids.append(0)
            attention_mask.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, input_type_ids, attention_mask

    
    def __getitem__(self, idx):
        audio_id = torch.tensor(self.audio_ids[idx], dtype=torch.long)

        #input_audio, original_audio = self.get_audio(idx)
        input_audio = self.get_audio(idx)

        text_input_ids, text_input_type_ids, text_attention_mask = self.get_text_input(
            idx
        )

        idx = torch.tensor(idx)

        return (
            input_audio,
            text_input_ids,
            text_attention_mask,
            idx,
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/audiocaption.yaml"
    

def encode_single_text(text, tokenizer_name="PORTULAN/albertina-ptpt-base", max_seq_length=128):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    input_ids = tokenizer.encode(text, max_length=max_seq_length, truncation=True)

    attention_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        attention_mask.append(0)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    return input_ids, attention_mask