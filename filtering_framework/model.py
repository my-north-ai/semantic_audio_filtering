import numpy as np

import torch
from torch import nn
from transformers import CLIPTextModel
from audio_encoder import WhisperForAudioClassification
from text_encoder import DebertaForSequenceClassification


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def weighted_loss(logits, sentence_sim, k=0.01):
    batch_size = logits.size(0)
    mask = 1 - torch.eye(batch_size).to(device=logits.device)

    sentence_sim = (sentence_sim * mask).mean(-1)

    normed_sim = sentence_sim / sentence_sim.sum()
    weight = torch.exp(normed_sim / k)

    labels = torch.arange(len(logits), device=logits.device)
    loss = weight * nn.functional.cross_entropy(logits, labels, reduction="none")
    loss = loss.sum() / weight.sum()

    return loss


def clip_loss(similarity: torch.Tensor, sentence_sim=None, type_loss="clip") -> torch.Tensor:
    if sentence_sim is not None and type_loss == "weighted_clip":
        text_loss = weighted_loss(similarity, sentence_sim)
        audio_loss = weighted_loss(similarity.T, sentence_sim)
    else:
        text_loss = contrastive_loss(similarity)
        audio_loss = contrastive_loss(similarity.T)
    return (text_loss + audio_loss) / 2.0


class MusCALL(nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_config = config.audio
        text_config = config.text

        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size

        self.do_audio_ssl = audio_config.ssl.do_ssl
        self.audio_ssl_loss_weight = (
            audio_config.ssl.ssl_loss_weight if self.do_audio_ssl else 0
        )

        self.type_loss = config.loss

        self.temperature = config.temperature

        if config.audio.model == "Whisper":
            self.audio_backbone = WhisperForAudioClassification.from_pretrained("openai/whisper-medium")
            self.audio_backbone.freeze_encoder()
        
      
        if config.text.model == "CLIPTextModel":
            pretrained_model = config.text.pretrained
            self.textual_head = CLIPTextModel.from_pretrained(pretrained_model)
        elif config.text.model == "Albertina":
            pretrained_model = config.text.pretrained
            self.textual_head = DebertaForSequenceClassification.from_pretrained(pretrained_model)
        elif config.text.model == "AlbertinaScratch":
            pretrained_model = config.text.pretrained
            self.textual_head = DebertaForSequenceClassification.from_pretrained(pretrained_model)
            for param in self.textual_head.deberta.encoder.parameters():
                param.requires_grad = False
            for param in self.textual_head.deberta.embeddings.parameters():
                param.requires_grad = False


        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        #if self.do_audio_ssl:
        #    print("Running audio SSL")
        #    self.audio_ssl = SimCLRAudio(
        #        encoder=self.audio_backbone,
        #        audio_config=audio_config,
        #    )

    def encode_audio(self, audio):
        hidden_states = self.audio_backbone(audio)[0]
        
        hidden_states = self.audio_projection(hidden_states)

        audio_features = hidden_states.mean(dim=1)

        return audio_features

    def encode_text(self, text, text_mask):
        if isinstance(self.textual_head, CLIPTextModel):
            outputs = self.textual_head(text, text_mask)
            pooled_outout = outputs.pooler_output
        elif isinstance(self.textual_head, DebertaForSequenceClassification):
            outputs = self.textual_head(input_ids=text, attention_mask=text_mask)
            pooled_outout = outputs[0]

        text_features = self.text_projection(pooled_outout)

        return text_features

    def forward(
        self,
        original_mel_spectograms,
        text,
        original_audio=None,
        sentence_sim=None,
        text_mask=None,
        return_loss=True,
    ):
        if return_loss:
            audio_ssl_loss = 0
            
  
        text_features = self.encode_text(text, text_mask)
        audio_features = self.encode_audio(original_mel_spectograms)

        # normalise features
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.temperature is None:
            # exponential of the logit scale
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 1.0 / self.temperature

        # audio features shape: batch_size x projection_dim
        # text features shape: batch_size x projection_dim  
        # logtis per audio shape: batch_size x batch_size
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()

        if return_loss:
            multimodal_loss = clip_loss(
                logits_per_text, sentence_sim, type_loss=self.type_loss
            )
            clip_loss_weight = 1 - self.audio_ssl_loss_weight
            loss = (multimodal_loss * clip_loss_weight) + (
                audio_ssl_loss * self.audio_ssl_loss_weight
            )

            return loss
        else:
            return logits_per_audio, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"