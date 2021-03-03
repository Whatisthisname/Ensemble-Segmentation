# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from init import Options
import monai
# from networks import build_net
import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.engines import (
    EnsembleEvaluator,
    SupervisedEvaluator,
    SupervisedTrainer
)
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, CheckpointSaver, LrScheduleHandler, CheckpointLoader, SegmentationSaver
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.utils import set_determinism

from monai.data import create_test_image_3d, list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (Activationsd,MeanEnsembled, LoadImaged, AsChannelFirstd, VoteEnsembled, AsDiscreted, Compose, AddChanneld, Transpose, ConcatItemsd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined,KeepLargestConnectedComponentd,
    Spacingd, Orientationd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

from monai.visualize import plot_2d_or_3d_image

def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device(opt.gpu_id)

    # ------- Data loader creation ----------

    # images
    images = sorted(glob(os.path.join(opt.images_folder, 'image*.nii')))
    segs = sorted(glob(os.path.join(opt.labels_folder, 'label*.nii')))

    train_files = []
    val_files = []

    for i in range(opt.models_ensemble):
        train_files.append(
            [
                {"image": img, "label": seg}
                for img, seg in zip(
                images[: (opt.split_val * i)] + images[(opt.split_val * (i + 1)): (len(images)-opt.split_val)],
                segs[: (opt.split_val * i)] + segs[(opt.split_val * (i + 1)): (len(images)-opt.split_val)])
            ]
        )
        val_files.append(
            [
                {"image": img, "label": seg}
                for img, seg in zip(
                images[(opt.split_val * i): (opt.split_val * (i + 1))],
                segs[(opt.split_val * i): (opt.split_val * (i + 1))])
            ]
        )

    test_files = [{"image": img, "label": seg}
                  for img, seg in zip(images[(len(images)-opt.split_test):len(images)],
                                            segs[(len(images)-opt.split_test):len(images)])]

    # ----------- Transforms list --------------

    if opt.resolution is not None:
        train_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                           padding_mode="zeros"),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
            RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),
            ToTensord(keys=['image', 'label'])
        ]
    else:
        train_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                           padding_mode="zeros"),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
            RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0, 0.3), prob=0.1),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            ToTensord(keys=['image', 'label'])
        ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    # ---------- Creation of DataLoaders -------------

    train_dss = [CacheDataset(data=train_files[i], transform=train_transforms) for i in range(opt.models_ensemble)]
    train_loaders = [DataLoader(train_dss[i], batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=torch.cuda.is_available())
                     for i in range(opt.models_ensemble)]

    val_dss = [CacheDataset(data=val_files[i], transform=val_transforms)
               for i in range(opt.models_ensemble)]
    val_loaders = [DataLoader(val_dss[i], batch_size=1, num_workers=opt.workers, pin_memory=torch.cuda.is_available())
                   for i in range(opt.models_ensemble)]

    test_ds = CacheDataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=opt.workers, pin_memory=torch.cuda.is_available())

    def train(index):

        # ---------- Build the nn-Unet network ------------

        if opt.resolution is None:
            sizes, spacings = opt.patch_size, opt.spacing
        else:
            sizes, spacings = opt.patch_size, opt.resolution

        strides, kernels = [], []


        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])

        net = monai.networks.nets.DynUNet(
            spatial_dims=3,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            res_block=True,
            # act=act_type,
            # norm=Norm.BATCH,

        ).to(device)

        from torch.autograd import Variable
        from torchsummaryX import summary

        data = Variable(
            torch.randn(int(opt.batch_size), int(opt.in_channels), int(opt.patch_size[0]), int(opt.patch_size[1]),
                        int(opt.patch_size[2]))).cuda()

        out = net(data)
        summary(net, data)
        print("out size: {}".format(out.size()))

        # if opt.preload is not None:
        #     net.load_state_dict(torch.load(opt.preload))

        # ---------- ------------------------ ------------

        optim = torch.optim.Adam(net.parameters(), lr=opt.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9
        )

        loss_function = monai.losses.DiceCELoss(sigmoid=True)

        val_post_transforms = Compose(
            [Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold_values=True),
            # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1])
             ])

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointSaver(save_dir="./runs/", save_dict={"net": net}, save_key_metric=True),
        ]

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_loaders[index],
            network=net,
            inferer=SlidingWindowInferer(roi_size=opt.patch_size, sw_batch_size=opt.batch_size, overlap=0.5),
            post_transform=val_post_transforms,
            key_val_metric={
                "val_mean_dice": MeanDice(
                    include_background=True,
                    output_transform=lambda x: (x["pred"], x["label"]),
                )
            },
            val_handlers=val_handlers
        )

        train_post_transforms = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True),
                # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            ])

        train_handlers = [
            ValidationHandler(validator=evaluator, interval=5, epoch_level=True),
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            StatsHandler(tag_name="train_loss",
                         output_transform=lambda x: x["loss"]),
            CheckpointSaver(save_dir="./runs/", save_dict={"net": net, "opt": optim}, save_final =True, epoch_level=True),
        ]

        trainer = SupervisedTrainer(
            device=device,
            max_epochs=opt.epochs,
            train_data_loader=train_loaders[index],
            network=net,
            optimizer=optim,
            loss_function=loss_function,
            inferer=SimpleInferer(),
            post_transform=train_post_transforms,
            amp=False,
            train_handlers=train_handlers,
        )
        trainer.run()
        return net

    models = [train(i) for i in range(opt.models_ensemble)]

    # -------- Test the models ---------

    def ensemble_evaluate(post_transforms, models):

        evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=test_loader,
            pred_keys=opt.pred_keys,
            networks=models,
            inferer=SlidingWindowInferer(
                roi_size=opt.patch_size, sw_batch_size=opt.batch_size, overlap=0.5),
            post_transform=post_transforms,
            key_val_metric={
                "test_mean_dice": MeanDice(
                    include_background=True,
                    output_transform=lambda x: (x["pred"], x["label"]),
                )
            },
        )
        evaluator.run()

    mean_post_transforms = Compose(
        [
            MeanEnsembled(
                keys=opt.pred_keys,
                output_key="pred",
                # in this particular example, we use validation metrics as weights
                weights=opt.weights_models,
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1])
        ]
    )

    print('Results from MeanEnsembled:')
    ensemble_evaluate(mean_post_transforms, models)

    vote_post_transforms = Compose(
        [
            Activationsd(keys=opt.pred_keys, sigmoid=True),
            # transform data into discrete before voting
            AsDiscreted(keys=opt.pred_keys, threshold_values=True),
            # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            VoteEnsembled(keys=opt.pred_keys, output_key="pred"),
        ]
    )

    print('Results from VoteEnsembled:')
    ensemble_evaluate(vote_post_transforms, models)


if __name__ == "__main__":
    main()
