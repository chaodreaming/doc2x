import glob
import os
import random
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from easydict import EasyDict
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_Transfer(model, cfg):
    saved = torch.load(cfg.Transfer, map_location=cfg.device)
    # model.load_state_dict(saved['state_dict'])
    old_state_dict = saved['state_dict']

    # 新建一个空的字典，用于存储新模型加载的权重
    new_state_dict = {}

    # 将旧模型中相同层的权重复制到新模型中
    all_layer = len(model.state_dict())
    num = 0
    for key in model.state_dict():
        if key in old_state_dict and old_state_dict[key].shape == model.state_dict()[key].shape:
            new_state_dict[key] = old_state_dict[key]
            num += 1
        else:
            new_state_dict[key] = model.state_dict()[key]

    # 使用load_state_dict()加载新模型的部分权重
    model.load_state_dict(new_state_dict)
    print("从预训练模型中加载了{}/{}层".format(num, all_layer))
    return model


def load_setting(setting):
    with open(setting, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(cfg)


def dir_check(path, name, increase=True):
    if increase == False:
        return os.path.join(path, name)
    if not os.path.exists(os.path.join(path, name)):
        os.makedirs(os.path.join(path, name))
        return os.path.join(path, name)
    else:

        i = 1
        while i:
            if not os.path.exists(os.path.join(path, name + str(i))):
                os.makedirs(os.path.join(path, name + str(i)))
                return os.path.join(path, name + str(i))
            i += 1


def find_latest_checkpoint(directory_pattern, file_pattern):
    """
    在给定的目录模式下找到最新的检查点文件。
    """
    directories = glob.glob(directory_pattern)
    if not directories:
        return None
    latest_dir = max(directories, key=os.path.getmtime)
    files = glob.glob(os.path.join(latest_dir, file_pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def pad(img: Image.Image, divable: int = 1) -> Image.Image:
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    threshold = 128
    data = np.array(img.convert("LA"))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255 - data[..., -1]).astype(np.uint8)

    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255 * (data < threshold).astype(np.uint8)
    else:
        gray = 255 * (data > threshold).astype(np.uint8)
        data = 255 - data

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b: b + h, a: a + w]
    im = Image.fromarray(rect).convert("L")
    dims: List[Union[int, int]] = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable * (div + (1 if mod > 0 else 0)))

    padded = Image.new("L", tuple(dims), 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


def alternatives(s):
    # TODO takes list of list of tokens
    # try to generate equivalent code eg \ne \neq or \to \rightarrow
    # alts = [s]
    # names = ['\\'+x for x in re.findall(ops, s)]
    # alts.append(re.sub(ops, lambda match: str(names.pop(0)), s))

    # return alts
    return [s]


def get_processor_model(model_dir, max_target_length, beam=False):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, ignore_mismatched_sizes=True)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_new_tokens = max_target_length
    if beam:
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     if name in ['encoder.pooler.dense.bias', 'encoder.pooler.dense.weight','encoder.embeddings.position_embeddings',"decoder.model.decoder.embed_tokens.weight"]:
    #         param.requires_grad =True
    #         print(name)
    return processor, model


def prepare_imgs(imgs: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
    output_imgs = []
    for img in imgs:
        if isinstance(img, (str, Path)):
            img =pad(Image.open(img)).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        else:
            raise ValueError(f'Unsupported image type: {type(img)}')
        output_imgs.append(img)

    return output_imgs