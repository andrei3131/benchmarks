#!/usr/bin/env python

import moxing as mox

mox.file.copy_parallel('s3://modelarts-cnnorth1-learning-course/dataset/ILSVRC2012/ImageNet-1k_tfRecord', 's3://obs-mnist-ic/imagenet2')
print('Copy procedure is completed')