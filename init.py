import argparse
import os

class Options():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--images_folder', type=str, default='./Data_folder/images')
        parser.add_argument('--labels_folder', type=str, default='./Data_folder/labels')
        parser.add_argument('--preload', type=str, default=None)
        parser.add_argument('--gpu_id', type=str, default="cuda:0", help='gpu ids')
        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

        # dataset parameters
        parser.add_argument('--patch_size', default=(128, 128, 128), help='Size of the patches extracted from the image')
        parser.add_argument('--spacing', default=[1, 1, 1], help='Original Resolution')
        parser.add_argument('--resolution', default=None, help='New Resolution to be resamplesS')
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=1, type=int, help='Channels of the output')
        # need to add one code part to have more input channels

        # training parameters
        parser.add_argument('--epochs', default=200, help='Number of epochs')
        parser.add_argument('--split_test', default=3, help='Number of samples for testing')
        parser.add_argument('--split_val', default=3, help='Number of samples for validation in each ensemble training')
        parser.add_argument('--models_ensemble', default=5, help='Number of models to train for ensemble')
        parser.add_argument('--pred_keys', default=["pred0", "pred1", "pred2", "pred3", "pred4"],
                             help='Models names, equal to the number of models')
        parser.add_argument('--weights_models', default=[0.95, 0.94, 0.95, 0.94, 0.90], help='Weights of models')
        parser.add_argument('--lr', default=0.002, help='Learning rate')
        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        # if opt.gpu_ids != '-1':
        #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        return opt





