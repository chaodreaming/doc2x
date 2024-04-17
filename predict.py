import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from models.models import Simple_Latex_OCR
from utils import load_setting, get_processor_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="configs/lksnorm.yaml",
                        help="Experiment settings")
    parser.add_argument("--target", "-t", type=str,
                        # default="datas/random_str/test",
                        default="datas/test_files/",
                        # default="datas/pdf",
                        # default="fc/tests/test_files/2.png",
                        help="OCR target (image or directory)")
    parser.add_argument("--model_dir", type=str, default="runs/2023_base2")

    args = parser.parse_args()
    # args.temperature=0.00001
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    processor, init_model = get_processor_model(cfg.model_dir, cfg.max_seq_len)
    print("setting:", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device="cpu"
    # load

    model = Simple_Latex_OCR(cfg, init_model, processor=processor, is_train=False)
    # saved = torch.load(cfg.checkpoint, map_location=device)
    #
    # model.load_state_dict(saved['state_dict'])

    model.to(device)
    model.eval()

    # print(model.training)
    target = Path(cfg.target)
    if target.is_dir():
        target = list(target.glob("*.jpg")) + list(target.glob("*.png"))
    else:
        target = [target]
    target = sorted(target)
    # result = model.recognize_formula([str(Path(image_fn)) for image_fn in target],batch_size=4)
    # print(result)
    for image_fn in tqdm(target):
        print(image_fn)
        start = time.time()
        images = []
        image_fn = str(Path(image_fn))

        result = model.recognize_formula([image_fn])
        # print(formula)

        res = {"formula": result[0]["text"] if len(result) < 2 else "\\begin{array}{c}" + " \\\\ ".join(
            result["text"]) + "\end{array}", }

        print("{}[{}]sec | image_fn : \n{}".format(image_fn, time.time() - start, res["formula"]))
        print()
