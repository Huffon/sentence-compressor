# Sentence Compressor using Transformer

This repository contains Sentence Compressor API trained using Transformer architecture.

<br/>

## Requirements

- Pyhon >= 3.7
- fairseq >= 0.9.0

<br/>

## Usage
- To preprocess dataset, run following command:

```bash
bash preprocess.sh
```

- To train Transformer using preprocessed dataset, run following command:

```bash
python train.py
```

- To generate example sentence using [pre-trained model](), run following command:

```
wget MODEL
tag xvzf MODEL
python generate.py
```

<br/>

## References
- [Sentence Compression Dataset](https://github.com/google-research-datasets/sentence-compression)
- [fairseq](https://github.com/pytorch/fairseq)
