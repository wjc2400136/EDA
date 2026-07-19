# Data Preparation

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

Obtain the 1,000 PNG images from the NIPS 2017 ImageNet-Compatible Dataset
under its original distribution terms and place them in data/images/.

Expected layout:

~~~text
data/
|-- images/
|   |-- IMAGE_1.png
|   |-- ...
|-- labels.csv
~~~

The bundled labels.csv contains filename, label, and targeted_label columns
with zero-based ImageNet class indices. The principal untargeted transfer
experiment uses filename and label. Do not rename, resize, or recompress the
images; model wrappers apply the required preprocessing.
