import argparse
import os
import time

import torch
from tqdm import tqdm

from models.models import Simple_Latex_OCR
from utils import load_setting, seed_torch, get_processor_model, process_latex

seed_torch()
basic = r"""
\begin{array}{c}
%s \\ %s
\end{array}"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str,
                        default="configs/im2latex.yaml",
                        help="Experiment settings")

    parser.add_argument("--model_dir", type=str,
                        default="runs/*",

                        )

    args = parser.parse_args()
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    processor, init_model = get_processor_model(cfg.model_dir, cfg.max_seq_len)
    print("setting:", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device="cpu"
    model = Simple_Latex_OCR(cfg, init_model, processor=processor, is_train=False)
    model.to(device)
    model.eval()

    df = []
    for index, image_dir in enumerate(cfg.image_dir):
        if index > len(cfg.val_data) - 1:
            continue
        with open(cfg.val_data[index]) as f:

            for i in f.readlines():
                img_name, img_label, h, w = str(i).split("\t")
                df.append([os.path.join(image_dir, img_name), img_label])


    corret = 0
    num = 0
    for batch in tqdm(df[:100]):
        image_fn, img_label = batch
        img_label = model.post_process(img_label)
        start = time.time()
        result = model.recognize_formula([image_fn])
        # print(formula)

        res = {"formula": result[0]["text"] if len(result) < 2 else "\\begin{array}{c}" + " \\\\ ".join(
            result["text"]) + "\end{array}", }

        pre = res["formula"]

        if not pre == img_label:
            print("\n")
            print("correctness", pre == img_label)

            print(time.time() - start, basic % (img_label, "\n" + pre))
        if pre == img_label:
            corret += 1
        num += 1
        print("acc ：{}".format(corret / num))
        print("*" * 100)
