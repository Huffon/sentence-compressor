"""Generation using Transformer"""
import os
import argparse


def generate(config):
    gen_prompt = f"fairseq-interactive {config.data_path} --path {config.ckpt} --input output/input.txt " \
                 f"--beam {config.beam} --print-alignment > output/output.txt " \
                 f"--temperature 0.9 --bpe gpt2"
    os.system(gen_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data-bin')
    parser.add_argument('--ckpt', type=str, default='ckpt/checkpoint_best.pt')
    parser.add_argument('--beam', type=int, default=5)
    args = parser.parse_args()
    generate(args)
