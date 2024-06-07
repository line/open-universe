# PLCMOS

*The code and data in this folder were released as part of the [PLC Challenge](https://github.com/microsoft/PLC-Challenge) under MIT Licence.*

This part of the repository contains the code and model weights for the PLCMOS non-intrusive metric for evaluating the performance of Packet Loss Concealment algorithms.

Install the packages from `requirements.txt` to use the model as part of your python evaluation scripts or notebooks (recommended), or `requirements_standalone.txt` to use plc_mos.py as a standalone evaluation tool.
     
Usage example:

```python
import soundfile as sf
from plc_mos import PLCMOSEstimator
plcmos = PLCMOSEstimator()
    
data, sr = sf.read("/some/wave/file/output/of/a/packet/loss/concealment/model.wav")
mos = plcmos.run(data, sr)
``` 

Usage example (standalone):

```shell
$ python plc_mos.py --degraded "example_wavs/*.wav"
100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.29it/s]
           filename_degraded  plcmos_v2
0     example_wavs\clean.wav   4.497762
1  example_wavs\plc_good.wav   4.123963
2  example_wavs\plc_poor.wav   2.740757
3.7874938382042784
$
```
