from fairseq.models.bart import BARTModel


def main():
    bart = BARTModel.from_pretrained('/checkpoints', checkpoint_file='model.pt')
    bart.eval()


if __name__ == "__main__":
    main()
