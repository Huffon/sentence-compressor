# Sentence Compressor

This repository contains Sentence Compressor API trained using **Transformer** and **BART** architecture

Lots of code are borrowed from [fairseq](https://github.com/pytorch/fairseq) library

<br/>

## Requirements

- **Python** version >= 3.7
- [PyTorch](https://pytorch.org/get-started/locally/) version >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0

```bash
conda create -n compression python=3.7
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install fairseq
```

<br/>

## Usage
- To **download** and **preprocess** dataset, run following command:

```bash
bash preprocess.sh
```

### (1) Transformer
- To train **Transformer** using pre-processed dataset, run following command:

```bash
python train_transformer.py
```

- To **generate** example sentence using [pre-trained Transformer](), run following command:

```
wget MODEL
tag xvzf tran.tar.gz
python generate_transformer.py
```

### (2) BART

- To fine-tune pre-trained **BART**, run following command:

```bash
bash train_bart.sh
```

- To **generate** example sentence using [fine-tuned BART](), run following command:

```
wget MODEL
tag xvzf bart.tar.gz
python generate_bart.py
```

<br/>

## Example

- To test your own sentences, fill [**input.txt**](output/input.txt) with your sentences

```
[Transformer]

Country music is itself a commercial offshoot of the traditional song and instrumental music of the South, formerly carried on in an oral tradition.
>> Country music is itself a commercial offshoot of the song.

General division has the most number of reviews, and Initmates division has the least number of reviews.
>> General division has the number of reviews.

[BART]

Country music is itself a commercial offshoot of the traditional song and instrumental music of the South, formerly carried on in an oral tradition.
>> Country music is itself a commercial offshoot of the traditional song.

General division has the most number of reviews, and Initmates division has the least number of reviews.
>> General division has the most number of reviews has the least number.
```

<br/>

## Data statistics

```
Source length
    Max: 4770 
    Min: 23 
    Avg: 153

Target length
    Max: 236
    Min: 6
    Avg: 56
```

<br/>

## References
- [**Sentence Compression Dataset**](https://github.com/google-research-datasets/sentence-compression)
- [**fairseq**](https://github.com/pytorch/fairseq)
- [fairseq Transformer __*train*__ script](https://github.com/kakaobrain/helo_word/blob/master/gec/track.py#L91)
- [Pre-trained **BART**](https://github.com/pytorch/fairseq/tree/master/examples/bart)