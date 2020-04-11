"""Data pre-processing function"""
import logging
from typing import List


def preprocess(mode: str, prefix: str, nums: List[str]):
    """Preprocess JSON containing parallel sentence compression data
    """
    logging.info(f"[{mode.upper()}]")
    src, tgt = list(), list()

    for i, num in enumerate(nums):
        logging.info(f"{i+1}th JSON is being pre-processed...")
        tmp_src, tmp_tgt = list(), list()

        with open(f"{prefix}{num}.json", "r") as f_json:
            lines = f_json.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('"sentence"'):
                    idx = line.find(":")
                    tmp_src.append(line[idx + 3 : -2])
                elif line.startswith('"text"'):
                    idx = line.find(":")
                    tmp_tgt.append(line[idx + 3 : -2])

            assert len(tmp_src) == len(tmp_tgt), "Source and Target datasets should be parallel!"

            tmp_src = tmp_src[::2]
            tmp_tgt = tmp_tgt[::2]

        src += tmp_src
        tgt += tmp_tgt

        logging.info(f"Current dataset size: {len(src)}")

    return src, tgt


def create_pair(mode: str, src: List[str], tgt: List[str]):
    """Create source and target file using pre-generated list
    """
    with open(f"data/{mode}.src", "w", encoding="utf-8") as f_src:
        for source in src:
            f_src.write(source)
            f_src.write("\n")

    with open(f"data/{mode}.tgt", "w", encoding="utf-8") as f_tgt:
        for target in tgt:
            f_tgt.write(target)
            f_tgt.write("\n")


def main():
    """Main function"""
    prefix = "data/sent-comp.train"
    nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    src, tgt = preprocess("train", prefix, nums)
    create_pair("train", src, tgt)

    prefix = "data/comp-data.eval"
    nums = [""]
    src, tgt = preprocess("val", prefix, nums)
    create_pair("eval", src, tgt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
