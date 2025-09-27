"""Define some utilities."""
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import transformers
from colorama import Fore, Style, init
from PIL import ImageDraw, ImageFont
from torch import nn
from tqdm import tqdm

from rec.predict.re_classifier import REClassifier
from rec.refer.refer import REFER
from rec.settings import TRANSFORMER_MODEL

init()
__color_table__ = {
    None: Style.RESET_ALL,
    "red": Fore.LIGHTRED_EX,
    "green": Fore.LIGHTGREEN_EX,
    "blue": Fore.LIGHTBLUE_EX,
}


def weight_init(m):
    """Define the weight init function."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def get_tokenizer(cache=None):
    """Get the pre-trained tokenizer."""
    if cache is None:
        return transformers.BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    model_path = os.path.join(cache, TRANSFORMER_MODEL)
    os.makedirs(model_path, exist_ok=True)

    if os.path.exists(os.path.join(model_path, "config.json")):
        return transformers.BertTokenizer.from_pretrained(model_path)

    tokenizer = transformers.BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    tokenizer.save_pretrained(model_path)

    return tokenizer


def conv3x3(in_channels, out_channels, num_groups=0):
    """Build a conv 3x3."""
    return nn.Sequential(
        # Conv2d w/o bias since BatchNorm2d/GroupNorm already accounts for it (affine=True)
        nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(out_channels)
        if num_groups < 1
        else nn.GroupNorm(num_groups, out_channels),
        nn.ReLU(inplace=True),
    )


def cprint(*parg, **kwargs):
    """Print color."""
    color = kwargs["color"] if "color" in kwargs else None
    print(__color_table__[color], end="")
    print(*parg, end="")
    print(Style.RESET_ALL)


def hms():
    """Get a specific format time based on time.time() funct."""
    return time.strftime("%H:%M:%S", time.gmtime(time.time()))


def progressbar(x, **kwargs):
    """Get a tqdm progressbar."""
    return tqdm(x, ascii=True, **kwargs)


def draw_bounding_boxes(
    img, bboxes, labels=None, fmt="xywh", color=(223, 223, 0), line_width=1
):
    """Draw a bounding box given a specific image."""
    assert fmt in ("xywh", "xyxy")
    line_width = line_width
    fnt_size = 8
    # fnt = ImageFont.truetype("arial.ttf", fnt_size)
    fnt = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fnt_size
    )

    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes):
        if fmt == "xywh":
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1]
        draw.rectangle(bbox, fill=None, outline=color, width=line_width)
        if labels is None:
            continue
        lbl = labels[i]
        x, y = bbox[0] + 1, bbox[1] + 1
        w, h = fnt.getsize(lbl)
        draw.rectangle((x, y, x + w, y + h), fill=color)
        draw.text((x, y), lbl, font=fnt, fill="black")
    del draw

    return img


def load_data(jsonl_file):
    """Load data from a json file."""
    data = []
    with gzip.open(jsonl_file, "rb") as fin:
        for line in fin:
            line = line.decode("utf-8")
            game = json.loads(line.strip("\n"))
            data.append(game)
    return data


def save_data(data, jsonl_file):
    """Save data into json file."""
    with gzip.open(jsonl_file, "wb") as fout:
        for x in data:
            json_bytes = (json.dumps(x) + "\n").encode("utf-8")
            fout.write(json_bytes)


def game2image(data, game_id):
    """Define a game2image function."""
    return [game["image"]["id"] for game in data if game["id"] == game_id][0]


def image2game(data, image_id):
    """Define a image2game function."""
    return [game["id"] for game in data if game["image"]["id"] == image_id][0]


def xyxy2xywh(boxes, inplace=False, as_int=False):
    """Convert boxes format: (x1, y1, x2, y2) -> (x, y, w, h).

    Args:
      boxes: input boxes in (x1, y1, x2, y2) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the +1 offset when computing the box width and height)
    Returns:
      boxes in (x, y, w, h) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + int(as_int)
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + int(as_int)
    return boxes


def xywh2xyxy(boxes, inplace=False, as_int=False):
    """Convert boxes format: (x, y, w, h) -> (x1, y1, x2, y2).

    Args:
      boxes: input boxes in (x, y, w, h) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the -1 offset when computing the box x2 and y2 coords)
    Returns:
      boxes in (x1, y1, x2, y2) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - int(as_int)
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - int(as_int)
    return boxes


def get_rec_counts(df_results):
    """Get some stats based on prediction results."""
    counts = {"hits": [], "counts": [], "rec_cls": []}
    for rec_cls in ["spatial", "ordinal", "relational", "intrinsic"]:
        counts["rec_cls"].append(rec_cls)
        mask_rec_cls = df_results[rec_cls] == 1
        counts["hits"].append(df_results.loc[mask_rec_cls, "hits"].sum())
        counts["counts"].append(df_results.loc[mask_rec_cls, "hits"].shape[0])
    return pd.DataFrame().from_dict(counts)


def get_rec_clf_counts(split="val", re_backend="stanza"):
    """Get counts for a berkeley dataset, based on train, valid, test. Returns a dataframe with the basic stats."""
    refer = REFER(
        "code/rec/refer/data",
        "refclef",
        "berkeley",
    )
    ref_ids = refer.getRefIds(split=split)
    all_, intrinsic, spatial, ordinal, relational = [], [], [], [], []
    classifier = REClassifier(backend=re_backend)
    for rid in progressbar(ref_ids):
        ref = refer.Refs[rid]
        sentences = [s["sent"] for s in ref["sentences"]]
        for i, sent in enumerate(sentences):
            stype = classifier.classify(sent)
            len_ = len(sent.split())
            all_.append(len_)
            if sum(stype) == 0:
                intrinsic.append(len_)
            if stype[0]:
                spatial.append(len_)
            if stype[1]:
                ordinal.append(len_)
            if stype[2]:
                relational.append(len_)
    d_to_frame = {
        "all": [len(all_), np.mean(all_), np.std(all_)],
        "intrinsic": [len(intrinsic), np.mean(intrinsic), np.std(intrinsic)],
        "spatial": [len(spatial), np.mean(spatial), np.std(spatial)],
        "ordinal": [len(ordinal), np.mean(ordinal), np.std(ordinal)],
        "relational": [len(relational), np.mean(relational), np.std(relational)],
    }
    df = pd.DataFrame().from_dict(d_to_frame).T.reset_index()
    df.columns = ["class", "count", "mean_sentence_length", "std_sentence_length"]
    return df
