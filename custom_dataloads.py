import argparse
import gc
import os.path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor

from utils import load_setting
from utils import seed_torch, pad

seed_torch()


class Latex_Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, txt_filename, processor):
        self.cfg = cfg
        self.images = []
        self.data = []
        self.processor = processor
        self.bos_token = self.processor.tokenizer.bos_token
        self.eos_token = self.processor.tokenizer.eos_token
        self.flag = False if len(self.processor.tokenizer) > 6400 else True

        print("data loading")

        skip_cnt, token_cnt = 0, 4
        for index, image_dir in tqdm(enumerate(self.cfg.image_dir)):

            with open(txt_filename[index]) as f:
                df = f.readlines()
            # if 1:
            #     # random.shuffle(df)
            #     df=df[:len(df)//100]

            for i in tqdm(df):
                try:
                    info = str(i).split("\t")
                    filename = info[0]
                    text = info[1]

                except ValueError:
                    print(ValueError)
                    skip_cnt += 1
                    continue

                if cfg.max_seq_len < len(text) + 4 or len(text) < cfg.min_seq_len:
                    skip_cnt += 1
                    continue
                # img exist
                if os.path.exists(os.path.join(image_dir, filename)) == False:
                    skip_cnt += 1
                    continue

                self.data.append([os.path.join(image_dir, filename), text])
        self.len = len(self.data)

        print(f"{self.len} data loaded. ({skip_cnt} data skipped)")
        del df, filename, text
        gc.collect()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_path, text = self.data[idx]

        image = pad(Image.open(file_path), 1).convert("RGB")
        if image.size[0] == 1 or image.size[1] == 1:
            # size 1 will err
            image = pad(Image.open(file_path), 2).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.cfg.max_seq_len).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


class CustomCollate(object):
    def __init__(self):
        pass

    def __call__(self, batchs):
        images = [batch["pixel_values"] for batch in batchs if batch["pixel_values"] is not None]
        texts = [batch["labels"] for batch in batchs if batch["labels"] is not None]

        if images and texts:
            return {
                "pixel_values": torch.cat(images).float(),
                "labels": torch.cat(texts)
            }
        else:
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="configs/im2latex.yaml",
                        help="Experiment settings")
    parser.add_argument("--name", type=str, default="exp",
                        help="Train experiment version")

    parser.add_argument("--num_workers", "-nw", type=int, default=10,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="Batch size for training and validate")

    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    processor = TrOCRProcessor.from_pretrained(cfg.model_dir)
    train_set = Latex_Dataset(cfg, cfg.train_data, processor)
    custom_collate = CustomCollate()
    train_data = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        # collate_fn=custom_collate,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        drop_last=True
    )
    # import psutil
    # process = psutil.Process(os.getpid())
    # for _ in range(10):
    #     for batch in tqdm(train_data):
    #         img, text = batch
    #         mm_info = process.memory_full_info()
    #         print(_,mm_info.uss / 1024 / 1024, "MB")
    #         break

    for _ in range(10):
        for batch in tqdm(train_data):
            img = batch["pixel_values"]
            text = batch["labels"]
