"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel


def main():
    bart = BARTModel.from_pretrained('ckpt_bart', checkpoint_file='checkpoint_best.pt')
    bart.cuda()
    bart.half()
    bart.eval()

    with open('output/input.txt') as source:
        lines = [line.replace("\n", "") for line in source.readlines()]
        
        with torch.no_grad():
            preds = bart.sample(lines, beam=4, lenpen=2.0, no_repeat_ngram_size=2, temperature=0.9)
            for i, (line, pred) in enumerate(zip(lines, preds)):
                print(f"[ori] ({i+1}): {line}")
                print(f"[com] ({i+1}): {pred}")
                print()


if __name__ == "__main__":
    main()
