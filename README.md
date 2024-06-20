# open-universe

This repository contains the code for the [UNIVERSE](https://arxiv.org/abs/2206.03065) and [UNIVERSE++](https://arxiv.org/abs/2406.12194)
universal speech enhancement models.

Audio samples of the models on various test sets are available [here](https://fakufaku.github.io/interspeech2024-universepp-samples/).

## Quick Start

Setup the environment with conda.
```bash
conda env create -f environment.yaml
conda activate open-universe
python -m pip install .
```

### Show me how to enhance some files!

That should be easy.
Here's how to do it
```bash
python -m open_universe.bin.enhance input/folder output/folder
```
This will pull the model from huggingface and enhance all the wav files in `input/folder`.

The API can be called from Python as follows.
```python
import torch
import torchaudio
from open_universe import inference_utils

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load some speech file for enhancement
noisy_speech, fs = torchaudio.load("path/to/audio.wav")

# load enhancement model (from checkpoint file or HF repo)
model = inference_utils.load_model("path/to/weights.ckpt", device=device)

# we can check the sampling frequency
if fs != model.fs:
    # throw an error, resampling the input would be better
    raise ValueError(f"Audio file and model do not use the same sampling frequency")

with torch.no_grad():
    enhanced_speech = model.enhance(noisy_speech.to(device))
```

### I want to train a model

You can easily train a toy model on [Voicebank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) as follows.
```bash
conda activate open-universe

# (optional) prepare the data
./data/prepare_voicebank_demand.sh

# train the model (UNIVERSE++, Voicebank-DEMAND, 16 kHz)
# this the default called if experiment is not specified
python ./train.py experiment=universepp_vb_16k

# train the model (UNIVERSE++, Voicebank-DEMAND, 24 kHz)
python ./train.py experiment=universepp_vb_24k

# train the model (UNIVERSE, Voicebank-DEMAND, 16 kHz)
python ./train.py experiment=universe_original_vb_16k
```
The config, experiment logs, and checkpoints are stored in `exp/<experiment>/<date-time>/`.
Training can be monitored with tensorboard by running.
```bash
tensorboard --logdir exp
```
Once training is done, you can evaluate your model, e.g. on the Voicebank-DEMAND test set
```bash
# run the enhancement on the noisy files and save the enhanced files
# along the other experiment data
python -m open_universe.bin.enhance \
    --model exp/default/2024-03-26_18-37-34_/checkpoints/step-00350000_score-0.0875.ckpt \
    data/voicebank_demand/16k/test/noisy \
    exp/default/2024-03-26_18-37-34_/results/step-00350000/vb-test-16k

# compute the metrics
python -m open_universe.bin.eval_metrics \
    exp/default/2024-03-26_18-37-34_/results/step-00350000/vb-test-16k \
    --ref_path data/voicebank_demand/16k/test/clean \
    --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext
```

You can print a table with results from different models.
```
python -m open_universe.bin.make_table \
    --format github \
    --results exp/universepp_vb_16k/2024-04-09_16-36-48_/results/step-00360000/vb-test-16k_summary.json \
              exp/universe_original_vb_16k/2024-03-29_13-51-38_/results/step-00300000/vb-test-16k_summary.json \
    --labels UNIVERSE++ UNIVERSE
```
This script was used to create the table in the following results section.


## Results

### Voicebank-DEMAND 16kHz

| model      |   si-sdr |   pesq-wb |   stoi-ext |   lsd |   lps |   OVRL |   SIG |   BAK |
|------------|----------|-----------|------------|-------|-------|--------|-------|-------|
| UNIVERSE++ |   18.624 |     3.017 |      0.864 | 4.867 | 0.937 |  3.200 | 3.489 | 4.040 |
| UNIVERSE   |   17.600 |     2.830 |      0.844 | 6.318 | 0.920 |  3.157 | 3.457 | 4.013 |

- `si-sdr`: scale-invariant signal-to-distortion ratio [Le Roux et al. 2018](https://arxiv.org/abs/1811.02508)
- `pesq-wb`: PESQ wideband [Rix et al. 2001](https://ieeexplore.ieee.org/abstract/document/941023)
- `stoi-ext`: Extended Short-Time Objective Intelligibility [Jensen and Taal 2016](https://ieeexplore.ieee.org/abstract/document/7539284)
- `lsd`: Log-Spectral Distance [Gray and Markel 1976](https://ieeexplore.ieee.org/abstract/document/1162849)
- `lps`: Levenshtein Phoneme Similarity [Pirklbauer et al. 2023](https://ieeexplore.ieee.org/abstract/document/10363040)
- `OVRL/SIG/BAK`: DNSMOS neural-based MOS prediction [Reddy et al. 2022](https://arxiv.org/abs/2110.01763)


## Contributing

Pull requests are welcome. In particular, we are interested in the following
contributions: bug fixes, new training configs, training code for new datasets.
For more details, see our [contribution guidelines](CONTRIBUTING.md).

## Citation

If you consider using this code for your own research, please cite our paper.
```latex
@inproceedings{universepp,
    authors={Scheibler, Robin and Fujita, Yusuke and Shirahata, Yuma and Komatsu, Tatsuya},
    title={Universal Score-based Speech Enhancement with High Content Preservation},
    booktitle={Proc. Interspeech 2024},
    month=sep,
    year=2024
}
```

## License

2024 Copyright LY Corporation

License: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
