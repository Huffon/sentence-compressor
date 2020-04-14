# Sentence Compressor

This repository contains **Sentence Compressor** API trained using **Transformer** and **BART** architecture

Lots of code are borrowed from [fairseq](https://github.com/pytorch/fairseq) library

<br/>

## Requirements

- **Python** version >= 3.7
- [PyTorch](https://pytorch.org/get-started/locally/) version >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0

```bash
conda create -n compressor python=3.7
conda activate compressor
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install fairseq requests pandas tensorflow-datasets
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

- To **generate** example sentence using pre-trained **Transformer**, run following command:

```
python generate_transformer.py
```

*cf. I think **Transformer** is not big enough to be trained on large datasets like Gigaword (4M). It is recommended to train Transformer only for extract-based datasets.*

<br/>

### (2) BART

- To **download** and **fine-tune** pre-trained **BART**, run following command:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar xvzf bart.large.tar.gz
bash train_bart.sh
```

- To **generate** example sentence using fine-tuned **BART**,, run following command:

```
python generate_bart.py
```

<br/>

## Example

- To **test** your own sentences, fill [**input.txt**](output/input.txt) with your sentences

```
# BART (fine-tuned 2000 steps; w/ Gigaword)

[Before & After]
Bartholomew JoJo Simpson is a fictional character in the American animated television series The Simpsons and part of the Simpson family.
>> Bartholomew JoJo Simpson is a fictional character in the animated television series The Simpsons.

Bart's most prominent and popular character traits are his mischievousness, rebelliousness and disrespect for authority.
>> Bart 's character traits are his mischievousness, rebelliousness and disrespect.

In casting, Nancy Cartwright originally planned to audition for the role of Lisa, while Yeardley Smith tried out for Bart.
>> Nancy Cartwright planned to audition for the role of Lisa.

Hallmarks of the character include his chalkboard gags in the opening sequence; his prank calls to Moe; and his catchphrases "Eat my shorts", "¡Ay, caramba!", "Don't have a cow, man!", and "I'm Bart Simpson. Who the hell are you?".
>> Hallmarks of the character include his chalkboard gags and his catchphrases.


# BART (fine-tuned 600 steps; w/o Gigaword)

[Before & After]
Bartholomew JoJo Simpson is a fictional character in the American animated television series The Simpsons and part of the Simpson family.
>> Bartholomew JoJo Simpson is a fictional character in the American animated television series The Simpsons.

Bart's most prominent and popular character traits are his mischievousness, rebelliousness and disrespect for authority.
>> Bart's most prominent character traits are his mischievousness, rebelliousness and disrespect for authority.

In casting, Nancy Cartwright originally planned to audition for the role of Lisa, while Yeardley Smith tried out for Bart.
>> Nancy Cartwright planned to audition for the role of Lisa, while Yeardley Smith tried for Bart.

Hallmarks of the character include his chalkboard gags in the opening sequence; his prank calls to Moe; and his catchphrases "Eat my shorts", "¡Ay, caramba!", "Don't have a cow, man!", and "I'm Bart Simpson. Who the hell are you?".
>> Hallmarks of the character include his chalkboard gags in the opening sequence.
```

<br/>

## Data statistics

```
Source length
    Max: 4770 
    Min: 4 
    Avg: 180

Target length
    Max: 477
    Min: 2
    Avg: 52
```

<br/>

## References
- [**Sentence Compression Dataset**](https://github.com/google-research-datasets/sentence-compression)
- [Pre-trained **BART**](https://github.com/pytorch/fairseq/tree/master/examples/bart)
- [**fairseq**](https://github.com/pytorch/fairseq)
- [fairseq Transformer __*train*__ script](https://github.com/kakaobrain/helo_word/blob/master/gec/track.py#L91)
- [Gigaword Dataset](https://www.tensorflow.org/datasets/catalog/gigaword) (Opt.)