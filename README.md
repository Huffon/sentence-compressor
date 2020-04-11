# Sentence Compressor using Transformer

This repository contains Sentence Compressor API trained using Transformer architecture.

<br/>

## Requirements

- Python version >= 3.7
- PyTorch version >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0

```bash
conda create -n compression python=3.7
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install fairseq
```

<br/>

## Usage
- To preprocess dataset, run following command:

```bash
bash preprocess.sh
```

- To train Transformer using preprocessed dataset, run following command:

```bash
python train_transformer.py
```

- To fine-tune pre-trained BART, run following command:

```
bash
bash train_bart.sh
```

- To generate example sentence using [pre-trained Transformer](), run following command:

```
wget MODEL
tag xvzf MODEL
python generate_transformer.py
```

- To generate example sentence using [fine-tuned BART](), run following command:

```
wget MODEL
tag xvzf MODEL
python generate_bart.py
```

<br/>

## References
- [**Sentence Compression Dataset**](https://github.com/google-research-datasets/sentence-compression)
- [**fairseq**](https://github.com/pytorch/fairseq)
- [fairseq __*train*__ script](https://github.com/kakaobrain/helo_word/blob/master/gec/track.py#L91)
- [Pre-trained **BART**](https://github.com/pytorch/fairseq/tree/master/examples/bart)