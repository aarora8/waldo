# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This module provides a pytorch-fashion customized dataset class
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from waldo.data_manipulation import convert_to_combined_image
from waldo.data_transformation import randomly_crop_combined_image


class Dataset_madcatar(Dataset):
    def __init__(self, dir, c_cfg, size, cache=True):
        self.c_cfg = c_cfg
        self.size = size
        self.dir = dir
        self.cache = cache
        self.data = []
        with open(self.dir + '/' + 'image_ids.txt', 'r') as ids_file:
            self.ids = ids_file.readlines()
        self.ids = [id.strip() for id in self.ids]
        # cache everything into memory if True
        if self.cache:
            for id in self.ids:
                image_with_mask = self.load_data(id)
                self.data.append(image_with_mask)

    def load_data(self, id):
        img_path = self.dir + '/numpy_arrays/' + id + '.img.npy'
        mask_path = self.dir + '/numpy_arrays/' + id + '.mask.npy'
        obj_class_path = self.dir + '/numpy_arrays/' + id + '.object_class.npy'

        image_with_mask = {}
        image_with_mask['img'] = np.load(img_path)
        image_with_mask['mask'] = np.load(mask_path)
        image_with_mask['object_class'] = np.load(obj_class_path).tolist()
        return image_with_mask

    def __getitem__(self, index):
        if self.cache:
            image_with_mask = self.data[index]
        else:
            id = self.ids[index]
            image_with_mask = self.load_data(id)
        combined_img = convert_to_combined_image(image_with_mask, self.c_cfg)
        n_classes = self.c_cfg.num_classes
        n_offsets = len(self.c_cfg.offsets)
        n_colors = self.c_cfg.num_colors
        cropped_img = randomly_crop_combined_image(
            combined_img, self.c_cfg, self.size, self.size)

        img = torch.from_numpy(
            cropped_img[:n_colors, :, :]).type(torch.FloatTensor)
        class_label = torch.from_numpy(
            cropped_img[n_colors:n_colors + n_classes, :, :]).type(torch.FloatTensor)
        bound = torch.from_numpy(
            cropped_img[n_colors + n_classes:n_colors +
                        n_classes + n_offsets, :, :]).type(torch.FloatTensor)

        return img, class_label, bound

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    from waldo.core_config import CoreConfig
    c_config = CoreConfig()
    c_config.read('exp/unet_5_10_sgd/configs/core.config')
    trainset = Dataset_madcatar('data/train',
                               c_config, 128, cache=False)
    trainloader = DataLoader(
        trainset, num_workers=1, batch_size=16, shuffle=True)
    data_iter = iter(trainloader)
    # data_iter.next()
    img, class_label, bound = data_iter.next()
    # torchvision.utils.save_image(class_label[:, 0:1, :, :], 'class0.png')
    # torchvision.utils.save_image(class_label[:, 1:2, :, :], 'class1.png')
    # torchvision.utils.save_image(bound[:, 0:1, :, :], 'bound0.png')
    # torchvision.utils.save_image(bound[:, 1:2, :, :], 'bound1.png')

