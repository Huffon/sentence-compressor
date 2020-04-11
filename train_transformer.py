"Training script"
import os
import argparse


def train(config):
    model_config = f"--arch transformer --share-all-embeddings " \
                   f"--optimizer adam --lr {config.lr} --label-smoothing 0.1 --dropout {config.dropout} " \
                   f"--max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                   f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                   f"--max-epoch {config.max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                   f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 "
    train_prompt = f"fairseq-train {config.data_path} {model_config} --save-dir {config.ckpt} "
    os.system(train_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data-bin')
    parser.add_argument('--ckpt', type=str, default='ckpt')
    args = parser.parse_args()
    train(args)