# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# Phoneme Loss

A phoneme prediction loss for speech enhancement

pre-trained model used for phoneme prediction:
https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft

This loss was inspired by the phoneme similarity metric proposed by
Pirklbauer et al., "Evaluation Metrics for Generative Speech Enhancement Methods: Issues and Perspectives," 2023.
Code for the metric: https://git.rz.tu-bs.de/ifn-public/LPS
"""
import torch
import torchaudio
from hydra.utils import instantiate
from torch.nn import Module
from transformers import Wav2Vec2ForCTC


class PhonemePredictor(Module):
    """
    This class uses a wav2vec2 based phone predictor to predict phonemes of
    an input waveform.

    Modified from https://git.rz.tu-bs.de/ifn-public/LPS
    """

    def __init__(self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft"):
        super().__init__()
        self.sr = 16000
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

        self.model.freeze_feature_encoder()
        for p in self.model.parameters():
            p.requires_grad = False

        self.eval()

    @property
    def blank(self):
        return self.model.config.pad_token_id

    def forward(self, waveform):
        # remove the channel dimension if it exists
        if waveform.ndim == 3:
            waveform = waveform[:, 0, :]

        # Here we replace Wav2Vec2Processor.__call__ by a simple normalization
        # this is because the original wav2vec2 processor is designed for feature
        # extraction and will break the autodiff
        m = waveform.mean(dim=-1, keepdim=True)
        v = waveform.var(dim=-1, keepdim=True)
        input_values = (waveform - m) / (v + 1e-7).sqrt()

        # retrieve logits
        logits = self.model(input_values).logits  # takes (batch, seq_len) input
        return logits


class PhonemeEmbeddingLoss(Module):
    """
    This loss uses a wav2vec2 based phone predictor to predict phonemes of the
    enhanced speech and the clean speech.
    Then, a cross-entropy loss is used with the predicted phonemes of the clean speech
    used as targets.
    """

    def __init__(
        self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft", sr=16000, loss=None
    ):
        super().__init__()
        self.sr = sr
        self.phoneme_predictor = PhonemePredictor(checkpoint)

        if loss is None:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = instantiate(loss, _recursive_=False)

        if self.sr != self.phoneme_predictor.sr:
            self.resampler = torchaudio.transforms.Resample(
                self.sr, self.phoneme_predictor.sr
            )
        else:
            self.resampler = lambda x: x

    def forward(self, input, target):
        # logits of the input signal
        input = self.resampler(input)
        input = self.phoneme_predictor(input)

        # process the targets to get the phoneme class predictions
        with torch.no_grad():
            target = self.resampler(target)
            target = self.phoneme_predictor(target)

        # return the embedding distance
        return self.loss(input, target)


class PhonemeLoss(Module):
    """
    This loss uses a wav2vec2 based phone predictor to predict phonemes of the
    enhanced speech and the clean speech.
    Then, a cross-entropy loss is used with the predicted phonemes of the clean speech
    used as targets.
    """

    def __init__(self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft", sr=16000):
        super().__init__()
        self.sr = sr
        self.phoneme_predictor = PhonemePredictor(checkpoint)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.sr != self.phoneme_predictor.sr:
            self.resampler = torchaudio.transforms.Resample(
                self.sr, self.phoneme_predictor.sr
            )
        else:
            self.resampler = lambda x: x

    def forward(self, input, target):
        # logits of the input signal
        input = self.resampler(input)
        input = self.phoneme_predictor(input)

        # process the targets to get the phoneme class predictions
        with torch.no_grad():
            target = self.resampler(target)
            target = self.phoneme_predictor(target)
            target = target.argmax(dim=-1)

        # cross entropy expects input (batch, num_classes) and (batch,)
        input = input.flatten(end_dim=-2)
        target = target.flatten()

        # return the cross-entropy
        return self.ce_loss(input, target)


class PhonemeCTCLoss(Module):
    """
    This loss uses a wav2vec2 based phone predictor to predict phonemes of the
    enhanced speech and the clean speech.
    Then, a cross-entropy loss is used with the predicted phonemes of the clean speech
    used as targets.
    """

    def __init__(self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft", sr=16000):
        super().__init__()
        self.sr = sr
        self.phoneme_predictor = PhonemePredictor(checkpoint)
        self.blank = self.phoneme_predictor.blank
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank)
        if self.sr != self.phoneme_predictor.sr:
            self.resampler = torchaudio.transforms.Resample(
                self.sr, self.phoneme_predictor.sr
            )
        else:
            self.resampler = lambda x: x

    def _targets_to_ctc(self, target):
        # process the targets to get the phoneme class predictions
        with torch.no_grad():
            target = self.resampler(target)
            target = self.phoneme_predictor(target)

            # decode to phoneme ids
            target = target.argmax(dim=-1)
            ids, lengths = [], []
            for i in range(target.shape[0]):
                dedup_seq = torch.unique_consecutive(target[i])
                mask_blank = dedup_seq != self.blank
                ids.append(dedup_seq[mask_blank])
                lengths.append(ids[-1].shape[0])

            # pad sequences to the same length
            S = max(lengths)
            ids = [
                torch.cat([i, i.new_full((S - i.shape[0],), self.blank)]) for i in ids
            ]
            ids = torch.stack(ids, dim=0)
            lengths = torch.tensor(lengths, device=ids.device)

        return ids, lengths

    def forward(self, input, target):
        # logits of the input signal
        input = self.resampler(input)
        input = self.phoneme_predictor(input)
        input = torch.nn.functional.log_softmax(input, dim=-1)

        # create the ctc targets
        target, target_lengths = self._targets_to_ctc(target)

        # CTC loss expects (seq_len, batch, num_classes) for the input
        input = input.transpose(0, 1).contiguous()
        # inputs are all the same lengths
        input_lengths = target_lengths.new_full((input.shape[1],), input.shape[0])

        # return the cross-entropy
        return self.ctc_loss(input, target, input_lengths, target_lengths)
