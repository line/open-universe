# MIT License
#
# Copyright (c) 2023 Jan Pirklbauer, Institut für Nachrichtentechnik, TU Braunschweig
# Modified by LY Corporation 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
# Levenshtein Phoneme Similarity (LPS)

Implementation of the LPS metric as described in:
> J. Pirklbauer, M. Sach, and K. Fluyt, Evaluation Metrics for Generative
> Speech Enhancement Methods: Issues and Perspectives. DE: VDE VERLAG GMBH,
> 2023. Accessed: Feb. 16, 2024. [Online]. Available:
> https://doi.org/10.30420/456164052

This code is a slightly modified version of the original code:
https://git.rz.tu-bs.de/ifn-public/LPS

It was modified as follows by Robin Scheibler
1. Resampling of input signal from any sampling frequency to 16 kHz
2. Numpy/Tensor cast for compatibility
"""
from typing import Union
import numpy as np
from Levenshtein import distance
import torch
import torchaudio
from torch.nn import Module
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

WAV2VEC2_SR = 16000


class PhonemePredictor(Module):
    """
    A simple wrapper around the Wav2Vec2 model to predict phonemes from an audio signal.

    Parameters
    ----------
    checkpoint : str
        The huggingface repo containing the checkpoint to use for the phoneme
        prediction model.
    """

    def __init__(self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
        self.sr = WAV2VEC2_SR

    def forward(self, waveform):
        """
        Parameters
        ----------
        waveform : torch.Tensor or np.ndarray
            The input waveform to predict phonemes from.

        Returns
        -------
        str
            The predicted phonemes.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        # maybe we can implement this part in torch...
        input_values = self.processor(
            waveform, return_tensors="pt", sampling_rate=self.sr
        ).input_values

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)


class LevenshteinPhonemeSimilarity:
    """
    Class to compute the Levenshtein phoneme similarity between two audio signals.
    If the sampling frequency of the audio signals to evaluate is different from that
    of the phoneme recognition model, the signals are resampled.

    Parameters
    ----------
    sr : int
        The sampling rate of the audio signals to evaluate
    """

    def __init__(self, sr=WAV2VEC2_SR):
        self.phoneme_predictor = PhonemePredictor()
        self.sr = sr

        if sr != self.phoneme_predictor.sr:
            self.resampler = torchaudio.transforms.Resample(
                sr, self.phoneme_predictor.sr
            )
        else:
            self.resampler = None

    def _resample(self, audio):
        if self.resampler is not None:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            audio = self.resampler(audio)
        return audio

    def __call__(
        self,
        sample: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute the Levenshtein phoneme similarity between two audio signals.

        Parameters
        ----------
        sample : torch.Tensor or np.ndarray
            The audio signal to evaluate.
        reference : torch.Tensor or np.ndarray
            The reference audio signal used to extract the reference phonemes.

        Returns
        -------
        float
            The Levenshtein phoneme similarity between the two audio signals.
            The similarity is just `LPS = 1 - phoneme_distance / len(ref_phonems)`,
            where `phoneme_distance` is the Levenshtein distance between the phonemes
            and `ref_phonems` is the phoneme string of the reference.
        """

        sample = self._resample(sample)
        reference = self._resample(reference)

        with torch.no_grad():
            sample_phonems = self.phoneme_predictor.forward(sample)[0].replace(" ", "")
            ref_phonems = self.phoneme_predictor.forward(reference)[0].replace(" ", "")
        lev_distance = distance(sample_phonems, ref_phonems)
        return 1 - lev_distance / len(ref_phonems)
