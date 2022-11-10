'''
detector-free referring expresion comprehension
'''
import os
import transformers
import torch

import pytorch_lightning as pl
from typing import Dict
from yaer.base import experiment_component

import rec.models as m
from rec.parser import ArgumentParser
from rec.utils import cprint, get_tokenizer
from rec.datasets import collate_fn, RefCLEF, RefCOCO, RefCOCOp, RefCOCOg, RegionDescriptionsVisualGnome
from rec.transforms import get_transform


@experiment_component
def get_datasets_splits(
    tokenizer, data_args: Dict = None,
    loss_args: Dict = None, runtime_args: Dict = None
    ) -> tuple:
    """Get datasets and splits based on the arguments."""
    if data_args["dataset"] == 'vg':
        vg = RegionDescriptionsVisualGnome(
            data_root='./VisualGnome',
            transform=get_transform('train', input_size=data_args["input_size"]),  # also for validation
            tokenizer=tokenizer,
            max_length=data_args["max_length"],
            with_mask_bbox=bool(loss_args["mu"] > 0.0),
        )
        n_train = int(0.9 * len(vg))
        n_val = max(0, len(vg) - n_train)
        datasets = torch.utils.data.random_split(
            vg, [n_train, n_val],
            generator=torch.Generator().manual_seed(runtime_args["seed"])
        )
        datasets = {'train': datasets[0], 'val': datasets[1]}
        ds_splits = ('train', 'val')
    else:
        if data_args["dataset"] == 'refclef':
            ds_class, ds_splits = RefCLEF, ('train', 'val', 'test')
        elif data_args["dataset"] == 'refcoco':
            ds_class, ds_splits = RefCOCO, ('train', 'val', 'testA', 'testB')
        elif data_args["dataset"] == 'refcoco+':
            ds_class, ds_splits = RefCOCOp, ('train', 'val', 'testA', 'testB')
        elif data_args["dataset"] == 'refcocog':
            ds_class, ds_splits = RefCOCOg, ('train', 'val', 'test')
        else:
            raise RuntimeError('invalid dataset')

        if runtime_args["debug"]:
            ds_splits = ds_splits[:2]  # train, val only

        datasets = {
            split: ds_class(
                split,
                transform=get_transform(split, input_size=data_args["input_size"]),
                tokenizer=tokenizer,
                max_length=data_args["max_length"],
                with_mask_bbox=bool(loss_args["mu"] > 0.0)
            ) for split in ds_splits
        }
    return datasets, ds_splits


@experiment_component
def base_experiment(
    model_args=None, loss_args=None, trainer_args=None,
    runtime_args=None
    ):
    num_workers = runtime_args["num_workers"]
    pl.seed_everything(runtime_args["seed"])
    transformers.logging.set_verbosity_error()

    tokenizer = get_tokenizer(runtime_args["cache"])
    datasets, ds_splits = get_datasets_splits(tokenizer)
    # data loaders
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=trainer_args["batch_size"],
            shuffle=bool(split == 'train') or bool(split == 'trainval'),
            num_workers=num_workers,
            pin_memory=bool(torch.cuda.is_available() and runtime_args["gpus"] is not None),
            collate_fn=collate_fn,
            drop_last=bool('test' not in split),
            persistent_workers=bool(num_workers > 0),
        ) for split in ds_splits
    }

    pdata = 0.02 if runtime_args["debug"] else 1.0

    model = m.IntuitionKillingMachine(
        backbone=model_args["backbone"],
        pretrained=True,
        num_heads=model_args["num_heads"],
        num_layers=model_args["num_layers"],
        num_conv=model_args["num_conv"],
        dropout_p=model_args["dropout_p"],
        segmentation_head=bool(loss_args["mu"] > 0.0),
        mask_pooling=model_args["mask_pooling"]
    )

    # learning rate scheduler
    scheduler_param = {}
    if trainer_args["scheduler"]:
        scheduler_param = {
            'milestones': [int(p * trainer_args["max_epochs"]) for p in (0.6, 0.9)],
            'gamma': 0.1
        }

    # model
    lit_model = m.LitModel(
        model=model,
        beta=loss_args["beta"],
        gamma=loss_args["gamma"],
        mu=loss_args["mu"],
        learning_rate=trainer_args["learning_rate"],
        weight_decay=trainer_args["weight_decay"],
        scheduler_param=scheduler_param
    )

    if runtime_args["checkpoint"] is not None:
        # continue training and logging on the same dir
        # WARNING: make sure you use the same model/trainer arguments
        output_dir = os.path.dirname(runtime_args["checkpoint"])
    else:
        # output dir from input arguments
        output_dir = ""
    os.makedirs(output_dir, exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir,
        name='',
        version='',
        default_hp_metric=False
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename='best',
        monitor='acc/val',
        mode='max',
        save_last=runtime_args["save_last"],
        verbose=False,
        every_n_epochs=1,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='acc/val',
        min_delta=0.0,
        patience=5,
        verbose=False,
        mode='max'
    )

    callbacks = [lr_monitor_callback, ]
    if not runtime_args["debug"]:
        callbacks.append(checkpoint_callback)
        if runtime_args["early_stopping"]:
            callbacks.append(early_stopping_callback)

    profiler = None
    if runtime_args["profile"]:
        profiler = pl.profiler.PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
        )

    gpus, strategy = None, None
    if runtime_args["gpus"] is not None:
        gpus = [int(i) for i in runtime_args["gpus"].split(',')]

        if not runtime_args["force_ddp"] and len(gpus) > 1:
            try:
                import fairscale
            except ModuleNotFoundError:
                raise ModuleNotFoundError('you need fairscale to train with multiple GPUs')
            strategy = pl.plugins.DDPShardedPlugin()
        else:
            strategy = pl.plugins.DDPPlugin(find_unused_parameters=True)

    trainer = pl.Trainer(
        profiler=profiler,
        gpus=gpus,
        max_epochs=trainer_args["max_epochs"],
        benchmark=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        strategy=strategy,
        limit_train_batches=pdata,
        limit_val_batches=pdata,
        accumulate_grad_batches=trainer_args["grad_steps"],
        enable_checkpointing=bool(not runtime_args["debug"]),
        precision=16 if runtime_args["amp"] else 32,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val'],
        ckpt_path=runtime_args["checkpoint"]
    )

    if runtime_args["debug"]:
        return

    for split in [s for s in ds_splits if s not in ('train', 'val')]:
        print(f'evaluating \'{split}\' split ...')
        trainer.test(
            dataloaders=loaders[split],
            ckpt_path=checkpoint_callback.best_model_path
        )
