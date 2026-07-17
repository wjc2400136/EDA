import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from pathlib import Path

os.environ.setdefault('TORCH_HOME', str(Path.home() / '.cache' / 'torch'))
import timm

if hasattr(timm.models, 'hub'):
    timm.models.hub.HUB_SERVER = os.environ.get(
        'TIMM_HUB_SERVER',
        'https://mirrors.tuna.tsinghua.edu.cn/pytorch/models/',
    )
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os


# ==============================================================================
# 新增：从 Normalize.py 中复制 TfNormalize 类的定义
# ==============================================================================
class TfNormalize(nn.Module):
    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


# ==============================================================================

# ==============================================================================
# 新增：创建一个 nn.Module 来调整张量大小
# ==============================================================================
class ResizeModule(nn.Module):
    def __init__(self, size):
        super(ResizeModule, self).__init__()
        self.resize_op = transforms.Resize(size)

    def forward(self, x):
        # 确保输入是 float 类型以进行插值
        return self.resize_op(x)


# ==============================================================================

img_height, img_width = 224, 224
img_max, img_min = 1., 0

# 修改后的模型列表
cnn_model_paper = [
    'resnet50_l2_eps1', 'resnet50_l2_eps5', 'vgg19', 'resnet18', 'resnet50', 'resnet101', 'resnext50_32x4d',
    'densenet121', 'mobilenet_v2',
    'inception_v3',
]

vit_model_paper = ['inception_v4', 'inception_resnet_v2',
                   'vit_base_patch16_224', 'deit_base_distilled_patch16_224', 'levit_256', 'pit_b_224', 'cait_s24_224',
                   'convit_base',
                   'tnt_s_patch16_224', 'visformer_small', 'swin_tiny_patch4_window7_224', 'coat_tiny',
                   ]  # 'adv_inception_v3','ens_adv_inception_resnet_v2',

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

# ==============================================================================
# 新增：定义要加载的 TensorFlow 模型列表
# ==============================================================================
tf_robust_models = [
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2'
]
# ==============================================================================

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]


def load_pretrained_model(cnn_model=[], vit_model=[], tf_robust_model=[]):
    for model_name in cnn_model:
        if model_name == "resnet50_l2_eps1":
            from .MadryLab import model_utils as Madry_model_utils
            from .MadryLab.datasets import ImageNet as Madry_ImageNet
            print('=> Loading model {} from robust-imagenet-models'.format(model_name))
            m, _ = Madry_model_utils.make_and_restore_model(
                arch='resnet50',
                dataset=Madry_ImageNet(''),
                resume_path="./robust-imagenet-models/resnet50_l2_eps1.ckpt"
            )
            yield model_name, wrap_model(m.model.eval().cuda())
        elif model_name == "resnet50_l2_eps5":
            from .MadryLab import model_utils as Madry_model_utils
            from .MadryLab.datasets import ImageNet as Madry_ImageNet
            print('=> Loading model {} from robust-imagenet-models'.format(model_name))
            m, _ = Madry_model_utils.make_and_restore_model(
                arch='resnet50',
                dataset=Madry_ImageNet(''),
                resume_path="./robust-imagenet-models/resnet50_l2_eps5.ckpt"
            )
            yield model_name, wrap_model(m.model.eval().cuda())
        else:
            yield model_name, wrap_model(models.__dict__[model_name](weights="DEFAULT").eval().cuda())
    for model_name in vit_model:
        yield model_name, wrap_model(timm.create_model(model_name, pretrained=True).eval().cuda())

    # ==============================================================================
    # 修改：加载 TensorFlow 模型的逻辑，增加尺寸调整
    # ==============================================================================
    for model_name in tf_robust_model:
        print(f'=> Loading TensorFlow model {model_name}')
        try:
            # 动态导入模型模块
            from torch_nets import (
                tf_adv_inception_v3,
                tf_ens3_adv_inc_v3,
                tf_ens4_adv_inc_v3,
                tf_ens_adv_inc_res_v2,
            )

            model_map = {
                'tf_adv_inception_v3': tf_adv_inception_v3,
                'tf_ens3_adv_inc_v3': tf_ens3_adv_inc_v3,
                'tf_ens4_adv_inc_v3': tf_ens4_adv_inc_v3,
                'tf_ens_adv_inc_res_v2': tf_ens_adv_inc_res_v2,
            }

            if model_name in model_map:
                net_class = model_map[model_name]
                model_path = os.path.join('./models/', model_name + '.npy')

                # 创建核心模型
                core_model = net_class.KitModel(model_path).eval().cuda()

                # InceptionV3 系列模型需要 299x299 的输入
                # 我们创建一个包含调整大小、归一化和核心模型的序列
                model = nn.Sequential(
                    ResizeModule(299),  # <--- 关键修改：使用自定义模块调整大小
                    TfNormalize('tensorflow'),
                    core_model
                )
                yield model_name, model
            else:
                print(f'[Warning] Model {model_name} not found in model_map. Skipping.')

        except ImportError:
            print(f"[Error] 'torch_nets' module not found. Cannot load {model_name}. Please ensure it's installed.")
        except FileNotFoundError:
            print(f"[Error] Model file not found for {model_name} at {model_path}. Skipping.")
        except Exception as e:
            print(f"[Error] An unexpected error occurred while loading {model_name}: {e}")
    # ==============================================================================


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    model_name = model.__class__.__name__
    Resize = 224

    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                                 dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
