# Google
git clone https://github.com/google-research-datasets/sentence-compression.git
mv sentence-compression/data data

# Edinburgh
wget https://www.jamesclarke.net/media/data/broadcastnews-compressions.tar.gz
tar xvzf broadcastnews-compressions.tar.gz 

wget https://www.jamesclarke.net/media/data/written-compressions.tar.gz
tar xvzf written-compressions.tar.gz

# MSR
wget https://download.microsoft.com/download/C/C/A/CCA10A93-4372-44CB-A2A3-B433ADB27276/Release.zip
mkdir Release
unzip Release.zip -d Release

rm broadcastnews-compressions.tar.gz written-compressions.tar.gz Release.zip

mkdir data
python utils/preprocess.py

rm -rf annotator1 annotator2 annotator3 written Release sentence-compression

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/$SPLIT.$LANG" \
    --outputs "data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref data/train.bpe \
    --validpref data/val.bpe \
    --destdir data-bin \
    --workers 60 \
    --srcdict dict.txt \
    --tgtdict dict.txt