"""
Predict runner.
Examples:
predict ~/my_thesis/rec/models/exp_003/best.ckpt --params ~/my_thesis/rec/models/exp_003/params.log --gpus 1
"""
import argparse
import os
import ast
import torch
import transformers
import pandas as pd
from torchvision.ops import box_iou

import rec.models as m
from rec.utils import cprint, progressbar, get_tokenizer, get_rec_counts
from rec.transforms import get_transform, undo_box_transforms_batch
from rec.datasets import collate_fn, RefCLEF, RefCOCO, RefCOCOp, RefCOCOg
from rec.predict.re_classifier import REClassifier


@torch.no_grad()
def iou(preds, targets):
    assert preds.size() == targets.size()
    preds = preds.unsqueeze(1)  # Nx1x4
    targets = targets.unsqueeze(1)  # Nx1x4
    return torch.FloatTensor([
        box_iou(preds[i], targets[i])
        for i in range(preds.size(0))
    ])


@torch.no_grad()
def process_batch(batch, model, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    batch['tok']['input_ids'] = batch['tok']['input_ids'].to(device)
    batch['tok']['attention_mask'] = batch['tok']['attention_mask'].to(device)

    preds, _ = model(batch)

    # to original coordinates
    preds = undo_box_transforms_batch(preds, batch['tr_param'])

    # clamp to original image size
    h0, w0 = batch['image_size'].unbind(1)
    image_size = torch.stack([w0, h0, w0, h0], dim=1)
    preds = torch.clamp(preds, torch.zeros_like(image_size), image_size-1)
    return preds


@torch.no_grad()
def test(model, loader, rec, iou_threshold=0.5):
    device = next(model.parameters()).device
    results = {
        "bbox_raw": [],
        "bbox_pred": [],
        "img_filename": [],
        "expr": [],
        "hits": [],
        "spatial": [],
        "ordinal": [],
        "relational": []
    }
    for batch in progressbar(loader, total=len(loader)):
        results["bbox_raw"].extend(batch["bbox_raw"])
        results["img_filename"].extend(batch["img_filename"])
        results["expr"].extend(batch["expr"])
        preds = process_batch(batch, model, device)
        results["bbox_pred"].extend(preds)
        iou_ = iou(preds, batch['bbox_raw'])
        hits = (iou_ > iou_threshold).float().detach().tolist()
        results["hits"].extend(hits)
        spatial, ordinal, relational = zip(*[
            rec.classify(expr) for expr in batch['expr']
        ])
        results["spatial"].extend(spatial)
        results["ordinal"].extend(ordinal)
        results["relational"].extend(relational)
    df_results = pd.DataFrame().from_dict(results)
    df_results.loc[:, "intrinsic"] = (
        df_results.loc[:, ["spatial", "ordinal", "relational"]].sum(axis=1) == 0
    ).astype(int)
    return df_results


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Detector-free grounding (test)',
        add_help=True,
        allow_abbrev=False
    )
    parser.add_argument(
        'checkpoint',
        help="trained model",
        type=str
    )
    parser.add_argument(
        '--params',
        help="trained model parameters. If not set, parameters will be read from the checkpoint file",
        type=str,
        default=None
    )
    parser.add_argument(
        '--max-length',
        help='if not set, read it from the checkpoint file',
        type=int
    )
    parser.add_argument(
        '--input-size',
        help='if not set, read it from the checkpoint file',
        type=int
    )
    parser.add_argument(
        '--iou-threshold',
        help='IOU threshold',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--batch-size',
        help='batch size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--gpus',
        help='GPU id',
        type=int
    )
    parser.add_argument(
        '--num-workers',
        help='dataloader num workers',
        type=int
    )
    parser.add_argument(
        '--get-sample',
        help='if set, run test script in a small batch of data. used for testing purposes.',
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--dump',
        help='if set, save REC results and other predictions info into a pandas csv.',
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    cprint(f'{vars(args)}', color='red')
    return args


def run():
    args = get_args()

    num_workers = 0 if args.num_workers is None else args.num_workers

    transformers.logging.set_verbosity_error()

    # ------------------------------------------------------------------------
    if args.params:
        # parse model arguments from a .log file
        with open(args.params) as l:
            for line in l:
                params = ast.literal_eval(line)
            dataset, max_length, input_size = params['dataset'], params['max_length'], params['input_size']
            backbone, num_heads, num_layers = params['backbone'], params['num_heads'], params['num_layers']
            num_conv, mu, mask_pooling = params['num_conv'], params['mu'], params['mask_pooling']
    else:
        # parse model arguments from checkpoint path
        exp_dirname = os.path.split(os.path.dirname(args.checkpoint))[1]
        _, _, dataset, max_length, input_size, backbone, num_heads, num_layers, num_conv, beta, gamma, mu, mask_pooling = exp_dirname.split('_')[:13]
        # the order of the remaining's filename are: (learning_rate, weight_decay, batch_size, grad_steps,
        # max_epochs, scheduler, early_stopping, amp, debug)
    max_length = int(max_length) if args.max_length is None else args.max_length
    input_size = int(input_size) if args.input_size is None else args.input_size
    num_layers = int(num_layers)
    num_heads = int(num_heads)
    num_conv = int(num_conv)
    segmentation_head = bool(float(mu) > 0.0)
    mask_pooling = bool(mask_pooling == '1')
    get_sample = args.get_sample
    dump_results = args.dump

    if torch.cuda.is_available() and args.gpus is not None:
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device('cpu')
    for ag in ["dataset", "max_length", "input_size", "backbone", "num_heads", "num_layers", "num_conv",
            "mu", "mask_pooling", "get_sample", "dump_results"]:
        print(f"Parameter: {ag}, value {vars()[ag]}")
    # ------------------------------------------------------------------------
    tokenizer = get_tokenizer()
    if dataset == 'refclef':
        ds_class, ds_splits = RefCLEF, ('val', 'test', )
    elif dataset == 'refcoco':
        ds_class, ds_splits = RefCOCO, ('val', 'testA', 'testB', )
    elif dataset == 'refcoco+':
        ds_class, ds_splits = RefCOCOp, ('val', 'testA', 'testB', )
    elif dataset == 'refcocog':
        ds_class, ds_splits = RefCOCOg, ('val', 'test', )
    else:
        raise RuntimeError('invalid dataset')
    datasets = {
        split: ds_class(
            split,
            transform=get_transform(split, input_size=input_size),
            tokenizer=tokenizer,
            max_length=max_length,
            with_mask_bbox=False,
            get_sample=get_sample
        ) for split in ds_splits
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,  # torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False,
        ) for split in ds_splits
    }
    model = m.IntuitionKillingMachine(
        backbone=backbone,
        pretrained=True,
        num_heads=num_heads,
        num_layers=num_layers,
        num_conv=num_conv,
        segmentation_head=segmentation_head,
        mask_pooling=mask_pooling
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage
    )
    # strip 'model.' from pl checkpoint
    state_dict = {
        k[len('model.'):]: v
        for k, v in checkpoint['state_dict'].items()
    }
    missing, _ = model.load_state_dict(state_dict, strict=False)
    # ensure the only missing keys are those of the segmentation head only
    assert [k for k in missing if 'segm' not in k] == []
    rec = REClassifier(backend='stanza', device=device)
    model.eval()
    for split in ds_splits:
        print(f'evaluating \'{split}\' split ...')
        df_results = test(model, loaders[split], rec)
        df_counts = get_rec_counts(df_results)
        if dump_results:
            df_results.to_parquet(
                args.checkpoint.replace("best.ckpt", f"predictions_{split}.parquet"), index=False
            )
            df_counts.to_csv(
                args.checkpoint.replace("best.ckpt", f"counts_{split}.csv", index=False)
            )
        print(df_counts)
