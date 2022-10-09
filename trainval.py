'''
detector-free referring expresion comprehension
'''
import os
import transformers
import torch

import pytorch_lightning as pl
from parser import ArgumentParser

import models as m
from utils import cprint
from datasets import collate_fn, RefCLEF, RefCOCO, RefCOCOp, RefCOCOg, RegionDescriptionsVisualGnome
from transforms import get_transform
from encoders import get_tokenizer


def run(args):

    pl.seed_everything(args.seed)

    num_workers = 0 if args.num_workers is None else args.num_workers

    transformers.logging.set_verbosity_error()

    # ------------------------------------------------------------------------

    tokenizer = get_tokenizer(args.cache)

    if args.dataset == 'vg':
        vg = RegionDescriptionsVisualGnome(
            data_root='./VisualGnome',
            transform=get_transform('train', input_size=args.input_size),  # also for validation
            tokenizer=tokenizer,
            max_length=args.max_length,
            with_mask_bbox=bool(args.mu > 0.0),
        )
        n_train = int(0.9 * len(vg))
        n_val = max(0, len(vg) - n_train)
        datasets = torch.utils.data.random_split(
            vg, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        datasets = {'train': datasets[0], 'val': datasets[1]}
        ds_splits = ('train', 'val')

    else:
        if args.dataset == 'refclef':
            ds_class, ds_splits = RefCLEF, ('train', 'val', 'test')
        elif args.dataset == 'refcoco':
            ds_class, ds_splits = RefCOCO, ('train', 'val', 'testA', 'testB')
        elif args.dataset == 'refcoco+':
            ds_class, ds_splits = RefCOCOp, ('train', 'val', 'testA', 'testB')
        elif args.dataset == 'refcocog':
            ds_class, ds_splits = RefCOCOg, ('train', 'val', 'test')
        else:
            raise RuntimeError('invalid dataset')

        if args.debug:
            ds_splits = ds_splits[:2]  # train, val only

        datasets = {
            split: ds_class(
                split,
                transform=get_transform(split, input_size=args.input_size),
                tokenizer=tokenizer,
                max_length=args.max_length,
                with_mask_bbox=bool(args.mu > 0.0)
            ) for split in ds_splits
        }

    # data loaders
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=bool(split == 'train') or bool(split == 'trainval'),
            num_workers=num_workers,
            pin_memory=bool(torch.cuda.is_available() and args.gpus is not None),
            collate_fn=collate_fn,
            drop_last=bool('test' not in split),
            persistent_workers=bool(num_workers > 0),
        ) for split in ds_splits
    }

    pdata = 0.05 if args.debug else 1.0

    model = m.IntuitionKillingMachine(
        backbone=args.backbone,
        pretrained=True,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_conv=args.num_conv,
        dropout_p=args.dropout_p,
        segmentation_head=bool(args.mu > 0.0),
        mask_pooling=args.mask_pooling
    )

    # learning rate scheduler
    scheduler_param = {}
    if args.scheduler:
        scheduler_param = {
            'milestones': [int(p * args.max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        }

    # model
    lit_model = m.LitModel(
        model=model,
        beta=args.beta,
        gamma=args.gamma,
        mu=args.mu,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_param=scheduler_param
    )

    if args.checkpoint is not None:
        # continue training and logging on the same dir
        # WARNING: make sure you use the same model/trainer arguments
        output_dir = os.path.dirname(args.checkpoint)
    else:
        # output dir from input arguments
        output_dir = ArgumentParser.args_to_path(args, (
            '--dataset',
            '--max-length',
            '--input-size',
            '--backbone',
            # '--language-model',
            # '--dropout-p',
            '--num-heads',
            '--num-layers',
            '--num-conv',
            '--beta',
            '--gamma',
            '--mu',
            '--mask-pooling',
            '--learning-rate',
            '--weight-decay',
            '--batch-size',
            '--grad-steps',
            '--max-epochs',
            '--scheduler',
            '--early-stopping',
            '--amp',
            '--debug',
        ), values_only=True)
    os.makedirs(output_dir, exist_ok=True)
    cprint(f'{output_dir}', color='blue')

    # log arguments for future reference
    with open(output_dir + '.log', 'w') as fh:
        fh.write(f'{vars(args)}')

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
        save_last=args.save_last,
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
    if not args.debug:
        callbacks.append(checkpoint_callback)
        if args.early_stopping:
            callbacks.append(early_stopping_callback)

    profiler = None
    if args.profile:
        profiler = pl.profiler.PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
        )

    gpus, strategy = None, None
    if args.gpus is not None:
        gpus = [int(i) for i in args.gpus.split(',')]

        if not args.force_ddp and len(gpus) > 1:
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
        max_epochs=args.max_epochs,
        benchmark=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        strategy=strategy,
        limit_train_batches=pdata,
        limit_val_batches=pdata,
        accumulate_grad_batches=args.grad_steps,
        enable_checkpointing=bool(not args.debug),
        precision=16 if args.amp else 32,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val'],
        ckpt_path=args.checkpoint
    )

    if args.debug:
        return

    for split in [s for s in ds_splits if s not in ('train', 'val')]:
        print(f'evaluating \'{split}\' split ...')
        trainer.test(
            dataloaders=loaders[split],
            ckpt_path=checkpoint_callback.best_model_path
        )


if __name__ == '__main__':
    parser = ArgumentParser('Detector-free grounding')
    parser.add_model_args()
    parser.add_data_args()
    parser.add_loss_args()
    parser.add_trainer_args()
    parser.add_runtime_args()
    args = parser.parse_args()
    cprint(f'{vars(args)}', color='red')

    run(args)
