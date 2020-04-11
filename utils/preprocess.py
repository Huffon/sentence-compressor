"""Data pre-processing function"""
import os
import logging
from typing import List

import pandas as pd


def preprocess_google(mode: str, prefix: str, nums: List[str]):
    """Preprocess JSON containing parallel sentence compression data
    """
    src, tgt = list(), list()

    for i, num in enumerate(nums):
        logging.info(f"{i+1}th JSON is being pre-processed...")
        tmp_src, tmp_tgt = list(), list()

        with open(f"{prefix}{num}.json", "r") as f_json:
            lines = f_json.readlines()
            for line in lines:
                line = line.strip()
                idx = line.find(":")
                if line.startswith('"sentence"'):
                    tmp_src.append(line[idx + 3 : -2])
                elif line.startswith('"text"'):
                    tmp_tgt.append(line[idx + 3 : -2])

            assert len(tmp_src) == len(tmp_tgt), "Source and Target datasets should be parallel!"

            tmp_src = tmp_src[::2]
            tmp_tgt = tmp_tgt[::2]

        src += tmp_src
        tgt += tmp_tgt

        logging.info(f"Current dataset size: {len(src)}")

    return src, tgt


def preprocess_edin(edin_dirs: List[str], source: List[str], target: List[str]):
    """Preprocess Edinburgh's parallel sentence compression dataset
    """
    for edin_dir in edin_dirs:
        for f_name in os.listdir(edin_dir):
            with open(os.path.join(edin_dir, f_name), "r", encoding="utf-8") as f_p:
                lines = f_p.readlines()

                for line in lines:
                    line = line.strip()

                    s = line.find('>')
                    e = line.rfind('<')

                    if line.startswith("<original"):
                        source.append(line[s+1:e])
                    elif line.startswith("<compressed"):
                        target.append(line[s+1:e])

    logging.info(f"Current dataset size: {len(source)}")

    return source, target


def preprocess_msr(source: List[str], target: List[str]):
    """Preprocess MSR parallel sentence compression dataset
    """
    msr = "Release/RawData"
    datasets = ["train", "valid", "test"]

    for dataset in datasets:
        dataframe = pd.read_csv(os.path.join(msr, f"{dataset}.tsv"), delimiter='\t', header=None, error_bad_lines=False)
        
        for item in dataframe[2].iteritems():
            splitted = item[1].split("|||")

            src, tgt = splitted[0], splitted[1]
            tgt = tgt.split("\t")[0]

            source.append(src.strip())
            target.append(tgt.strip())

    logging.info(f"Current dataset size: {len(source)}")

    return source, target


def create_pair(mode: str, src: List[str], tgt: List[str]):
    """Create source and target file using pre-generated list
    """
    with open(f"data/{mode}.source", "w", encoding="utf-8") as f_src:
        for source in src:
            f_src.write(source)
            f_src.write("\n")

    with open(f"data/{mode}.target", "w", encoding="utf-8") as f_tgt:
        for target in tgt:
            f_tgt.write(target)
            f_tgt.write("\n")


def main():
    """Main function"""
    logging.info("[TRAIN] Google Dataset")
    prefix = "data/sent-comp.train"
    nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    src, tgt = preprocess_google("train", prefix, nums)

    logging.info("[TRAIN] Edinburgh Dataset")
    edin_dirs = ["annotator1", "annotator2", "annotator3", "written"]
    src, tgt = preprocess_edin(edin_dirs, src, tgt)

    logging.info("[TRAIN] MSR Dataset")
    src, tgt = preprocess_msr(src, tgt)
    create_pair("train", src, tgt)

    logging.info("[EVAL] Google Dataset")
    prefix = "data/comp-data.eval"
    nums = [""]
    src, tgt = preprocess_google("val", prefix, nums)
    create_pair("val", src, tgt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
