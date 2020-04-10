import os
import argparse


def generate(config):
    gen_prompt = f"fairseq-genearte {config.data_path} --path {config.ckpt} " \
                 f"--beam {config.beam} --max-tokens {max_tokens} --print-alignment > result.txt"
    os.system(gen_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data-bin/')
    parser.add_argument('--ckpt', type=str, default='ckpt/')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=15)
    args = parser.parse_args()
    generate(args)
