"""Generation using Transformer"""
import os
import argparse


def generate(config):
    gen_prompt = f"fairseq-interactive {config.data_path} --path {config.ckpt} --input output/input.txt " \
                 f"--beam {config.beam} --print-alignment > output/output.txt " \
                 f"--temperature 0.9 --bpe gpt2 --source-lang source --target-lang target " \
                 f"--no-repeat-ngram-size 2 --lenpen 2.0"
    os.system(gen_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ckpt')
    parser.add_argument('--ckpt', type=str, default='ckpt/checkpoint_best.pt')
    parser.add_argument('--beam', type=int, default=4)
    args = parser.parse_args()
    generate(args)
