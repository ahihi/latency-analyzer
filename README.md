# latency analyzer

## installation

```
conda create --no-shortcuts -c conda-forge -n latency-analyzer python numpy sortednp tk matplotlib librosa
source activate latency-analyzer
```

if you want to use `time-receiver.py`, also run:

```
pip install python-osc
```

## usage

```
python latency-analyzer.py [wavfile]
```