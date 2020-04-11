"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel


def main():
    bart = BARTModel.from_pretrained('checkpoints', checkpoint_file='checkpoint1.pt')
    bart.cuda()
    bart.half()
    bart.eval()

    with open('output/input.txt') as source:
        lines = source.readlines()
        print("[Before]")
        for i, line in enumerate(lines):
            print(f"({i+1}): {line}")
        
        with torch.no_grad():
            preds = bart.sample(lines, beam=4, lenpen=2.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
            print("[After]")
            for i, pred in enumerate(preds):
                print(f"({i+1}): {pred}")


if __name__ == "__main__":
    main()
