'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import PIL
from torchsampler import ImbalancedDatasetSampler
import hotfix.transforms
import math
# torch.manual_seed(0)
from . import autoaugment

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}
lighting_param = 0.1

params_dict = {
    'imagenet': {
        'train_dir': os.path.expanduser('/data1/trang/ImageNet/train/'),
        'val_dir': os.path.expanduser('/data1/trang/ImageNet/val/'),
        'num_train_samples': 1281167,
        'num_val_samples': 50000,
        'num_classes': 1000,
    },
    'myimagenet100': {
        'train_dir': os.path.expanduser('~/data/myimagenet100/train/'),
        'val_dir': os.path.expanduser('~/data/myimagenet100/val/'),
        'num_train_samples': 129395,
        'num_val_samples': 5000,
        'num_classes': 100,
    },
    'vgg-face2': {
        'train_dir': os.path.expanduser('/data1/trang/VGG-Face2/data/train'),
        'val_dir': os.path.expanduser('/data1/trang/VGG-Face2/data/test'),
        'num_train_samples': 3067564,
        'num_val_samples': 243722,
        'num_classes': 9131,
    },
    'cifar10': {
        'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/pytorch_cifar10'),
        'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/pytorch_cifar10'),
        'num_train_samples': 50000,
        'num_val_samples': 10000,
        'num_classes': 10,
    },

    'cifar100': {
        'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/pytorch_cifar100'),
        'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/pytorch_cifar100'),
        'num_train_samples': 50000,
        'num_val_samples': 10000,
        'num_classes': 100,
    },
    'rafdb':{
        'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS_FER/data/raf_db/train'),
        'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS_FER/data/raf_db/valid'),
        'num_train_samples': 12271,
        'num_val_samples': 3068,
        'num_classes': 7,
    },
    'AffectNet':{
        'train_dir': os.path.expanduser('/home/trangpi/Downloads/dataset/AffectNet/train'),
        'val_dir': os.path.expanduser('/home/trangpi/Downloads/dataset/AffectNet/valid'),
        'num_train_samples': 283901,
        'num_val_samples': 3500,
        'num_classes': 7,
    },
    'AffectNet8':{
    'train_dir': os.path.expanduser('/home/trangpi/Downloads/dataset/AffectNet8/train'),
    'val_dir': os.path.expanduser('/home/trangpi/Downloads/dataset/AffectNet8/valid'),
    'num_train_samples': 287651,
    'num_val_samples': 3999,
    'num_classes': 8,
    },
    'PlantVillage':{
    'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/plantvillage/train'),
    'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/plantvillage/val'),
    'num_train_samples': 49189,
    'num_val_samples': 12297,
    'num_classes': 39,
    },
    'pest': {
        'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/ip102_v1.1/images/train'),
        'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS/data/ip102_v1.1/images/test'),
        'num_train_samples': 45095,
        'num_val_samples': 22619,
        'num_classes': 102,
    },
    'rafdb1': {
        'train_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS_FER/data/raf_db/valid'),
        'val_dir': os.path.expanduser('/home/trangpi/Project/ZenNAS_FER/data/raf_db/valid'),
        'num_train_samples': 3068,
        'num_val_samples': 3068,
        'num_classes': 7,
    }
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def load_imagenet_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                       dataloader_testing):
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC), transforms.CenterCrop(input_image_size),
                          transforms.ToTensor(), transforms_normalize]
    else:
        if auto_augment:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              autoaugment.ImageNetPolicy(),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        else:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.4, 0.4, 0.4),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        pass
        if random_erase:
            transform_list.append(hotfix.transforms.RandomErasing())
    pass

    transformer = transforms.Compose(transform_list)

    the_dataset = dataset_ImageFolderClass(data_dir, transformer)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }

def load_cifar_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                    dataloader_testing=False):

    transforms_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        if input_image_size > 32:
            transform_list = [transforms.Resize(input_image_size, interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = []

        transform_list += [transforms.ToTensor(), transforms_normalize]
    else:

        if input_image_size > 32:
            resize_image_size = round(input_image_size / 0.75)
            transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC)]
            transform_list += [transforms.RandomResizedCrop(input_image_size, scale=(0.8, 1.0),
                                                            interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = [transforms.RandomCrop(input_image_size, padding=4)]

        if auto_augment:
            autoaugment_policy = autoaugment.CIFAR10Policy()
            transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                               transforms.ToTensor(),
                               transforms_normalize]
        else:
            transform_list += [transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms_normalize]
        pass
        if random_erase:
            transform_list.append(hotfix.transforms.RandomErasing())
    pass

    transformer = transforms.Compose(transform_list)

    if dataset_name == 'cifar10':
        the_dataset = datasets.CIFAR10(root=data_dir, train=set_name=='train', download=True, transform=transformer)
    elif dataset_name == 'cifar100':
        the_dataset = datasets.CIFAR100(root=data_dir, train=set_name=='train', download=True, transform=transformer)
    else:
        raise ValueError('Unknown dataset_name=' + dataset_name)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }

def load_raf_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                    dataloader_testing=False, type_auto_augment='CIFA'):

    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        if input_image_size > 32:
            transform_list = [transforms.Resize((input_image_size,input_image_size), interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = []
        print("val: " , transform_list)
        transform_list += [transforms.ToTensor(), transforms_normalize]
        print("val: ", transform_list)
    else:

        if input_image_size > 32:
            resize_image_size = round(input_image_size / 0.75)
            transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC)]
            transform_list += [transforms.RandomResizedCrop(input_image_size, scale=(0.8, 1.0),
                                                            interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = [transforms.RandomCrop(input_image_size, padding=4)]

        if auto_augment:
            if type_auto_augment == "CIFA":
                autoaugment_policy = autoaugment.CIFAR10Policy()
                transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                                   transforms.ToTensor(),
                                   transforms_normalize]
            else:
                autoaugment_policy = autoaugment.ImageNetPolicy()
                transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                                   transforms.ToTensor(),
                                   Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                                   transforms_normalize]

        else:
            transform_list += [transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms_normalize]
        pass
        if random_erase:
            transform_list.append(hotfix.transforms.RandomErasing(scale=(0.02, 0.1)))
    pass

    transformer = transforms.Compose(transform_list)
    # train_dir = os.path.join(data_dir, 'train')
    if dataset_name == 'rafdb' or dataset_name == 'PlantVillage' or dataset_name == 'pest' or  dataset_name == 'rafdb1' :
        the_dataset = datasets.ImageFolder(data_dir, transform=transformer)
    else:
        raise ValueError('Unknown dataset_name=' + dataset_name)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass
    # g = torch.Generator()
    # g.manual_seed(0)
    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }

def load_affect_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                    dataloader_testing=False, type_auto_augment='CIFA'):

    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        if input_image_size > 32:
            transform_list = [transforms.Resize(input_image_size, interpolation=PIL.Image.BICUBIC), transforms.CenterCrop(input_image_size)]
        else:
            transform_list = []

        transform_list += [transforms.ToTensor(), transforms_normalize]

    else:
        if auto_augment:

            if input_image_size > 32:
                resize_image_size = round(input_image_size / 0.75)
                transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC)]
                transform_list += [transforms.RandomResizedCrop(input_image_size, scale=(0.8, 1.0),
                                                                interpolation=PIL.Image.BICUBIC)]
            else:
                transform_list = [transforms.RandomCrop(input_image_size, padding=4)]
            if type_auto_augment == 'CIFA':
                autoaugment_policy = autoaugment.CIFAR10Policy()
                transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                                   transforms.ToTensor(),
                                   transforms_normalize]
            else:
                autoaugment_policy = autoaugment.ImageNetPolicy()
                transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                                   transforms.ToTensor(),
                                   Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                                   transforms_normalize]

        else:
            transform_list = [transforms.Resize(input_image_size, interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        pass
        if random_erase:
            # transform_list.append(hotfix.transforms.RandomErasing())
            transform_list.append(transforms.RandomErasing(scale=(0.4, 0.4)))
    pass

    transformer = transforms.Compose(transform_list)
    # train_dir = os.path.join(data_dir, 'train')
    if dataset_name == 'AffectNet' or 'AffectNet8' or 'pest':
        the_dataset = datasets.ImageFolder(data_dir, transform=transformer)
    else:
        raise ValueError('Unknown dataset_name=' + dataset_name)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    # print("value of shuffle and setname:", (shuffle, set_name))

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        if set_name == "train":
            sampler = ImbalancedDatasetSampler(the_dataset)
            # print("sampler here.......", str(type(sampler)))
        elif world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)
        else:
            sampler = None

        pass
    pass
    if set_name == "train":
        # print("set sampler Imbalanced is successfully")
        # print("sampler", type(sampler))
        data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True, sampler=sampler,
                                                  drop_last=drop_last)
    else:
        data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True, sampler=sampler,
                                                  drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }



def _get_data_(dataset_name=None, set_name=None, batch_size=None, train_augment=False, random_erase=False, auto_augment=False,
             input_image_size=224, input_image_crop=0.875, rank=0, world_size=1, shuffle=False,
             num_workers=6, drop_last=False, dataset_ImageFolderClass=None, dataloader_testing=False, argv=None, auto_augment_type=None):

    if dataset_name in ['imagenet', 'myimagenet100', 'vgg-face2']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_imagenet_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                                  random_erase=random_erase, auto_augment=auto_augment,
                                  data_dir=data_dir,
                                  input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                                  world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=drop_last,
                                  dataset_ImageFolderClass=dataset_ImageFolderClass,
                                  dataloader_testing=dataloader_testing)

    if dataset_name in ['cifar10', 'cifar100']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_cifar_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                               random_erase=random_erase, auto_augment=auto_augment,
                               data_dir=data_dir,
                               input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                               world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                               num_workers=num_workers, drop_last=drop_last,
                               dataset_ImageFolderClass=dataset_ImageFolderClass,
                               dataloader_testing=dataloader_testing)
    ## load dataset raf_db and affect_net

    if dataset_name in ['rafdb', 'PlantVillage', 'pest', 'rafdb1']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_raf_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                               random_erase=random_erase, auto_augment=auto_augment,
                               data_dir=data_dir,
                               input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                               world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                               num_workers=num_workers, drop_last=drop_last,
                               dataset_ImageFolderClass=dataset_ImageFolderClass,
                               dataloader_testing=dataloader_testing, type_auto_augment=auto_augment_type)

    if dataset_name in ['AffectNet', 'AffectNet8']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_affect_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                               random_erase=random_erase, auto_augment=auto_augment,
                               data_dir=data_dir,
                               input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                               world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                               num_workers=num_workers, drop_last=drop_last,
                               dataset_ImageFolderClass=dataset_ImageFolderClass,
                               dataloader_testing=dataloader_testing, type_auto_augment=auto_augment_type)




def get_data(opt, argv):
    dataset_name = opt.dataset
    batch_size = opt.batch_size_per_gpu
    random_erase = opt.random_erase
    auto_augment = opt.auto_augment
    input_image_size = opt.input_image_size
    input_image_crop = opt.input_image_crop
    rank = opt.rank
    world_size = opt.world_size
    num_workers = opt.workers_per_gpu
    auto_augment_type = opt.auto_augment_type
    shuffle_train_data = opt.shuffle_train_data

    # check if independent training
    if opt.independent_training:
        rank = 0
        world_size = 1


    # load train set
    set_name = 'train'
    if opt.no_data_augment:
        train_augment = False
    else:
        train_augment = True
    shuffle = shuffle_train_data
    # print("shuffle....", shuffle)
    drop_last = True

    train_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
             input_image_size, input_image_crop, rank, world_size, shuffle,
             num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, auto_augment_type=auto_augment_type)


    # load val set
    set_name = 'val'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False

    val_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                    input_image_size, input_image_crop, rank, world_size, shuffle,
                                    num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, auto_augment_type=auto_augment_type)

    return {
        'train_loader' : train_dataset_info['data_loader'],
        'val_loader' : val_dataset_info['data_loader'],
        'train_sampler': train_dataset_info['sampler'],
        'val_sampler': val_dataset_info['sampler'],
    }







