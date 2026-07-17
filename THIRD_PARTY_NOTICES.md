# Third-Party Notices

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

This repository contains code derived from or designed to interoperate with
third-party projects. This notice supplements, but does not replace, the
license terms in [`LICENSE`](LICENSE).

## TransferAttack

The attack framework and portions of the attack implementations are derived
from the TransferAttack project by Trustworthy-AI-Group. The upstream copyright
notice and MIT License are retained in this repository.

The EDA implementation, revision experiment runners, reproducibility documents,
and associated evaluation utilities contain modifications and additions by
the manuscript authors.

## Runtime Dependencies

PyTorch, torchvision, timm, RobustBench, LPIPS, pytorch-grad-cam, OpenAI client
libraries, and other packages listed in the requirements files remain subject
to their own licenses. Installing this project does not change those terms.

## Datasets and Model Weights

ImageNet, the NIPS 2017 ImageNet-Compatible Dataset, ImageNet-V2, pretrained
model weights, and provider-hosted VLMs are not licensed by this repository and
are not redistributed. Users must obtain them from their authorized sources and
comply with the corresponding terms.

## Method Implementations

The repository includes implementations of published attack methods for
research comparison. Their papers are cited in the manuscript and source code.
Users redistributing individual implementations should also inspect the original
project repositories and licenses associated with those methods.

