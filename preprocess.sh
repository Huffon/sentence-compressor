python preprocess.py

fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref data/train --validpref data/eval --destdir data-bin
