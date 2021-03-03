#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from train import *
import argparse
from monai.inferers import sliding_window_inference
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default='./Data_folder/images/image0.nii')
parser.add_argument("--label", type=str, default='./Data_folder/labels/label0.nii')
parser.add_argument("--result", type=str, default='./Data_folder/', help='path to the .nii result to save')
parser.add_argument("--weights", type=str, default="./runs/net_key_metric*", help='network weights to load')
parser.add_argument('--spacing', default=[1, 1, 1], help='original resolution')
parser.add_argument('--resolution', default=None, help='New Resolution, if you want to resample')
parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64), help="Input dimension for the generator")
parser.add_argument('--gpu_ids', type=str, default="cuda:0", help='gpu ids')
parser.add_argument('--models_ensemble', default=5, help='Number of models to train for ensemble')
parser.add_argument('--pred_keys', default=["pred0", "pred1", "pred2", "pred3", "pred4"],
                    help='Models names, equal to the number of models')
parser.add_argument('--weights_models', default=[0.95, 0.94, 0.95, 0.94, 0.90], help='Weights of models')
args = parser.parse_args()


def segment(image, label, weights, spacing, resolution, patch_size, gpu_ids, models_ensemble, pred_keys, weights_models, result):

    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device(gpu_ids)

    if label is not None:
        files = [{"image": image, "label": label}]
    else:
        files = [{"image": image}]

    if label is not None:

        if resolution is not None:
            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=resolution, mode=('bilinear', 'nearest')),
                ToTensord(keys=['image', 'label'])
            ])
        else:
            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                ToTensord(keys=['image', 'label'])
            ])

    else:
        if resolution is not None:
            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),
                ToTensord(keys=['image'])
            ])
        else:
            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                ToTensord(keys=['image'])
            ])

    val_ds = CacheDataset(data=files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    # ---------- Build the nn-Unet network ------------

    if resolution is None:
        sizes, spacings = patch_size, spacing
    else:
        sizes, spacings = patch_size, resolution

    def load(index,model_files,val_loader, sizes, spacings):

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
            in_channels=1,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            res_block=True,
            # act=act_type,
            # norm=Norm.BATCH,

        ).to(device)

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointLoader(load_path=model_files[index], load_dict={"net": net}),
        ]

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_loader,
            network=net,
            val_handlers=val_handlers)
        evaluator.run()

        # ---------- ------------------------ ------------

        return net

    # models file path
    model_files = glob(weights)
    models = [load(i, model_files, val_loader, sizes, spacings) for i in range(models_ensemble)]

    # -------- Run the inference ---------

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        SegmentationSaver(
            output_dir=result,
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: output["pred"],
        )]

    def ensemble_evaluate(post_transforms, models):

        if label is not None:

            evaluator = EnsembleEvaluator(
                device=device,
                val_data_loader=val_loader,
                pred_keys=pred_keys,
                networks=models,
                inferer=SlidingWindowInferer(
                    roi_size=patch_size, sw_batch_size=1, overlap=0.5),
                post_transform=post_transforms,
                key_val_metric={
                    "test_mean_dice": MeanDice(
                        include_background=True,
                        output_transform=lambda x: (x["pred"], x["label"]),
                    )
                },
                val_handlers=val_handlers
            )
            evaluator.run()

        else:

            evaluator = EnsembleEvaluator(
                device=device,
                val_data_loader=val_loader,
                pred_keys=pred_keys,
                networks=models,
                inferer=SlidingWindowInferer(
                    roi_size=patch_size, sw_batch_size=1, overlap=0.5),
                post_transform=post_transforms,
                val_handlers=val_handlers
            )
            evaluator.run()

    mean_post_transforms = Compose(
        [
            MeanEnsembled(
                keys=pred_keys,
                output_key="pred",
                # in this particular example, we use validation metrics as weights
                weights=weights_models,
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1])
        ]
    )
    ensemble_evaluate(mean_post_transforms, models)


if __name__ == "__main__":

    segment(args.image, args.label, args.weights, args.spacing, args.resolution,
            args.patch_size, args.gpu_ids, args.models_ensemble, args.pred_keys, args.weights_models, args.result)











