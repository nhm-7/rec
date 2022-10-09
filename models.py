import torch
import io

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.ops import box_convert, box_iou
from torchvision.utils import draw_bounding_boxes, make_grid

import embeddings as emb
import encoders as enc
from encoders import weight_init
from utils import conv3x3
from losses import GIoULoss, FocalLoss, SoftDiceLoss
from transforms import undo_box_transforms_batch, denormalize
from transformers_pos import (
    XTransformerEncoder, TransformerEncoder, TransformerEncoderLayer,
)


class IntuitionKillingMachine(nn.Module):
    def __init__(self,
                 backbone='resnet50', pretrained=True, embedding_size=256,
                 num_heads=8, num_layers=6, num_conv=4, dropout_p=0.1,
                 segmentation_head=True, mask_pooling=True):
        super().__init__()

        if backbone.endswith('+tr'):
            self.vis_enc = enc.TransformerImageEncoder(
                backbone=backbone.rstrip('+tr'),
                out_channels=embedding_size,
                pretrained=pretrained,
            )

        elif backbone.endswith('+fpn'):
            self.vis_enc = enc.FPNImageEncoder(
                backbone=backbone.rstrip('+fpn'),
                out_channels=embedding_size,
                pretrained=pretrained,
                with_pos=False
            )
        else:
            self.vis_enc = enc.ImageEncoder(
                backbone=backbone,
                out_channels=embedding_size,
                pretrained=pretrained,
                with_pos=False
            )

        # freeze ResNet stem
        if 'resnet' in backbone:
            self.vis_enc.backbone.conv1.requires_grad = False
            self.vis_enc.backbone.conv1.eval()

        self.vis_pos_emb = emb.LearnedPositionEmbedding2D(
            embedding_dim=embedding_size
        )

        self.lan_enc = enc.LanguageEncoder(
            out_features=embedding_size,
            global_pooling=False,
            dropout_p=dropout_p
        )

        self.lan_pos_emb = emb.LearnedPositionEmbedding1D(
            embedding_dim=embedding_size
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # ---
        # CONV PRE-HEAD (NECK?)

        if num_conv > 0:
            self.pre_head = nn.Sequential(*[
                conv3x3(embedding_size, embedding_size) for _ in range(num_conv)
            ])
            self.pre_head.apply(weight_init)
        else:
            self.pre_head = nn.Identity()

        # ---
        # OUTPUT HEADS

        # box prediction
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 4, bias=True),
            nn.Sigmoid()
        )
        self.head.apply(weight_init)

        # box segmentation mask
        self.segm_head = None
        if segmentation_head:
            self.segm_head = nn.Sequential(
                nn.Conv2d(embedding_size, 1, (3, 3), 1, 1, bias=True),
                #nn.Sigmoid()
            )
            self.segm_head.apply(weight_init)

        # ---

        self.mask_pooling = bool(mask_pooling)

        if self.mask_pooling and self.segm_head is None:
            raise RuntimeError('mask pooling w/o a segmentation head does not makes sense')

        self.embedding_size = embedding_size

    def slow_param_ids(self, slow_visual_backbone=True, slow_language_backbone=True):
        ids = []

        if slow_visual_backbone:
            ids += [id(p) for p in self.vis_enc.backbone.parameters()]
            if hasattr(self.vis_enc, 'encoder'):  # +tr
                ids += [id(p) for p in self.vis_enc.encoder.parameters()]

        if slow_language_backbone:
            if isinstance(self.lan_enc, enc.LanguageEncoder):
                ids += [id(p) for p in self.lan_enc.language_model.parameters()]
            else:
                ids += [id(p) for p in self.lan_enc.embeddings.parameters()]

        return ids

    def flatten(self, x):
        N, D, H, W = x.size()
        x = x.to(memory_format=torch.channels_last)
        x = x.permute(0, 2, 3, 1).view(N, H*W, D)
        return x  # NxHWxD

    def unflatten(self, x, size):
        N, R, D = x.size()
        H, W = size
        assert R == H*W, 'wrong tensor size'
        x = x.permute(0, 2, 1).to(memory_format=torch.contiguous_format)
        x = x.view(N, D, H, W)
        return x  # NxDxHxW

    def forward(self, input):
        img, mask, tok = input['image'], input['mask'], input['tok']

        # ---
        # VISUAL EMBEDDINGS

        x, x_mask = self.vis_enc(img, mask)   # NxDxHxW, NxHxW
        x_pos = self.vis_pos_emb(x, x_mask)

        N, D, H, W = x.size()  # save dims before flatten

        x = self.flatten(x)  # NxRxD
        x_mask = self.flatten(x_mask).squeeze(-1)  # NxR
        x_pos = self.flatten(x_pos)   # NxRxD

        # ---
        # LANGUAGE EMBEDDINGS

        z, z_mask = self.lan_enc(tok)   # NxTxD, NxT
        z_pos = self.lan_pos_emb(z)  # NxTxD

        # ---
        # V+L TRANSFORMER

        # [...visual...]+[[CLS]...language tokens...[SEP]]
        xz = torch.cat([x, z], dim=1)
        xz_mask = torch.cat([x_mask, z_mask], dim=1)
        xz_pos = torch.cat([x_pos, z_pos], dim=1)

        xz = self.encoder(xz, src_key_padding_mask=(xz_mask==0), pos=xz_pos)  #, size=(H,W))

        # restore spatiality of visual embeddings after cross-modal encoding
        xz_vis = xz[:, :H*W, ...]
        xz_vis = self.unflatten(xz_vis, (H, W))

        x_mask = self.unflatten(x_mask.unsqueeze(-1), (H, W))

        # ---

        # convolutional pre-head
        xz_vis = self.pre_head(xz_vis)

        # ---

        # segmentation head w/ (opt.) pooling
        segm_mask, pooled_feat = None, None
        if self.segm_head is not None:
            segm_mask = torch.sigmoid(self.segm_head(xz_vis)) * x_mask
            if self.mask_pooling:  # box mask guided pooling
                pooled_feat = (segm_mask * xz_vis).sum((2, 3)) / segm_mask.sum((2, 3))
            segm_mask = F.interpolate(segm_mask, img.size()[2:], mode='bilinear', align_corners=True)

        # if not mask_pooling, do the pooling using all visual feats (equiv. to a uniform mask)
        if pooled_feat is None:
            pooled_feat = (x_mask * xz_vis).sum((2, 3)) / x_mask.sum((2, 3))

        # bbox prediction
        pred = self.head(pooled_feat)
        pred = box_convert(pred, 'cxcywh', 'xyxy')

        return pred, segm_mask


class LitModel(pl.LightningModule):
    def __init__(self, model, beta, gamma, mu, learning_rate, weight_decay,
                 scheduler_param):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.mu = mu
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l1_loss = nn.SmoothL1Loss(reduction='mean', beta=beta)
        self.giou_loss = GIoULoss(reduction='mean')
        self.segm_loss = FocalLoss(reduction='mean')
        self.scheduler_param = scheduler_param

    @torch.no_grad()
    def peep(self, batch, preds, idxs=[0,]):
        N, _, H, W = batch['image'].size()
        size = torch.tensor([W, H, W, H], device=preds.device)

        imlist = []
        for i in idxs:
            image = (255 * denormalize(batch['image'])[i]).byte()
            boxes = torch.stack([batch['bbox'][i], preds[i]], dim=0) * size
            img = draw_bounding_boxes(image.cpu(), boxes.cpu(), colors=['blue', 'red'])

            plt.imshow(img.permute(1, 2, 0))
            plt.title(batch['expr'][i])
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg', bbox_inches='tight')
            buf.seek(0)

            img = ToTensor()(Image.open(buf))
            imlist.append(
                torch.nn.functional.interpolate(img.unsqueeze(0), (320, 320), mode='bilinear').squeeze(0)
            )

        return imlist

    @torch.no_grad()
    def iou(self, preds, targets):
        assert preds.size() == targets.size()
        preds = preds.unsqueeze(1)  # Nx1x4
        targets = targets.unsqueeze(1)  # Nx1x4
        return torch.FloatTensor([
            box_iou(preds[i], targets[i])
            for i in range(preds.size(0))
        ])

    def loss(self, dbox, dmask):
        l1_loss = self.l1_loss(dbox['preds'], dbox['targets'])

        giou_loss = 0.0
        if self.gamma > 0.0:
            giou_loss = self.giou_loss(dbox['preds'], dbox['targets'])

        segm_loss = 0.0
        if dmask['targets'] is not None and self.mu > 0.0:
            segm_loss = self.segm_loss(dmask['preds'], dmask['targets'])

        loss = l1_loss + self.gamma * giou_loss + self.mu * segm_loss

        return loss, (l1_loss, giou_loss, segm_loss)

    def training_step(self, batch, batch_idx):
        preds, segm_mask = self.model(batch)
        # AMP
        preds = preds.to(batch['bbox'].dtype)
        if segm_mask is not None:
            segm_mask = segm_mask.to(batch['mask_bbox'].dtype)

        loss, loss_terms = self.loss(
            dbox={'preds': preds, 'targets': batch['bbox']},
            dmask={'preds': segm_mask, 'targets': batch['mask_bbox']}
        )

        l1_loss, giou_loss, segm_loss = loss_terms

        self.log('loss/train_l1', l1_loss.detach(), on_step=True, on_epoch=False)

        self.log('loss/train_giou', giou_loss.detach(), on_step=True, on_epoch=False)

        if segm_mask is not None and self.mu > 0.0:
            self.log('loss/train_segm', segm_loss.detach(), on_step=True, on_epoch=False)

        self.log('loss/train', loss.detach(), on_step=True, on_epoch=True)

        iou = self.iou(preds, batch['bbox'])
        self.log('iou/train', iou.mean().detach(), on_step=False, on_epoch=True)

        hits = (iou > 0.5).float()
        self.log('acc/train', hits.mean().detach(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, segm_mask = self.model(batch)
        # AMP
        preds = preds.to(batch['bbox'].dtype)
        if segm_mask is not None:
            segm_mask = segm_mask.to(batch['mask_bbox'].dtype)

        loss, _ = self.loss(
            dbox={'preds': preds, 'targets': batch['bbox']},
            dmask={'preds': segm_mask, 'targets': batch['mask_bbox']}
        )

        self.log('loss/val', loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 2:  # skip dryrun
            idxs = list(range(0, preds.size(0), max(1, preds.size(0)//16)))
            grid = make_grid(self.peep(batch, preds, idxs=idxs), nrow=len(idxs))
            self.logger.experiment.add_image(
                'validation', grid, global_step=self.current_epoch
            )
            self.logger.experiment.flush()
        # to original image coordinates
        preds = undo_box_transforms_batch(preds, batch['tr_param'])
        # clamp to original image size
        h0, w0 = batch['image_size'].unbind(1)
        image_size = torch.stack([w0, h0, w0, h0], dim=1)
        preds = torch.clamp(preds, torch.zeros_like(image_size), image_size-1)

        iou = self.iou(preds, batch['bbox_raw'])
        self.log('iou/val', iou.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.25).float()
        self.log('acc/val25', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.50).float()
        self.log('acc/val', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.75).float()
        self.log('acc/val75', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        preds, _ = self.model(batch)
        # AMP
        preds = preds.to(batch['bbox'].dtype)

        # to original coordinates
        preds = undo_box_transforms_batch(preds, batch['tr_param'])

        # clamp to original image size
        h0, w0 = batch['image_size'].unbind(1)
        image_size = torch.stack([w0, h0, w0, h0], dim=1)
        preds = torch.clamp(preds, torch.zeros_like(image_size), image_size-1)

        iou = self.iou(preds, batch['bbox_raw'])
        self.log('iou/test', iou.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.5).float()
        self.log('acc/test', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        return

    def configure_optimizers(self):
        slow_ids = self.model.slow_param_ids()

        slow_params = [
            p for p in self.parameters()
            if id(p) in slow_ids and p.requires_grad
        ]

        fast_params = [
            p for p in self.parameters()
            if id(p) not in slow_ids and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {'params': slow_params, 'lr': 0.1*self.learning_rate},
                {'params': fast_params},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.scheduler_param in (None, {}):
            return optimizer

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.scheduler_param['milestones'],
                gamma=self.scheduler_param['gamma']
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer, ], [scheduler, ]
