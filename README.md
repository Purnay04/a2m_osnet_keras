# Implement OSNet in keras


## Contents
* [Installation](#installation)
* [Usage](#usage)
* [Training](#training)
* [Convert Trained Model to Full Integer Quantization](#convert-trained-model-to-full-integer-quantization)

## Installtion
```
mkvirtualenv --python==python3.6.9 osnet_keras
workon osnet_keras

pip install -r requirements.txt
```

## Usage
1. Download the "Training images (Task 1 & 2)" and "Validation images (all tasks)" from the [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) download page](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads).  
For example, I put the untarred files at ${HOME}/ImageNet/ILSVRC2012/.
2. Untar the "train" and "val" files.  

   ```shell
   $ cd ${HOME}/ImageNet/ILSVRC2012
   $ mkdir train
   $ cd train
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_train.tar
   $ find . -name "*.tar" | while read NAME ; do \
         mkdir -p "${NAME%.tar}"; \
         tar -xvf "${NAME}" -C "${NAME%.tar}"; \
         rm -f "${NAME}"; \
     done
   $ cd ..
   $ mkdir validation
   $ cd validation
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_val.tar
   ```

3. Pre-process the validation image files.  (The script would move the JPEG files into corresponding subfolders.)

   ```shell
   $ cd data
   $ python3 ./preprocess_imagenet_validation_data.py \
             ${HOME}/ImageNet/ILSVRC2012/validation \
             imagenet_2012_validation_synset_labels.txt
    ```

4. Build TFRecord files for "train" and "validation".

   ```shell
   $ mkdir ${HOME}/ImageNet/ILSVRC2012/tfrecords
   $ python3 build_imagenet_data.py \
             --output_directory ${HOME}/ImageNet/ILSVRC2012/tfrecords \
             --train_directory ${HOME}/ImageNet/ILSVRC2012/train \
             --validation_directory ${HOME}/ImageNet/ILSVRC2012/validation


## Training
   ```shell
   $ ./train.sh osnet
   ```

7. Evaluate accuracy of the trained osnet model.

   ```shell
   $ python evaluate.py --dataset_dir ${HOME}/ImageNet/ILSVRC2012/tfrecords \
                         checkpoints/osnet-model-final.h5
   ```

## Convert Trained Model to Full Integer Quantization
This experiment uses post-training quantization  
> Use tf v1.15 or >= 2.3 for training  
> Use tf >= 2.3 for converting

[tutorial](https://colab.research.google.com/drive/1e1hPDxp9MGliRBba6zGVyBMJCeeo7Pho?usp=sharing)

## References

## Structure
```
.
├── build_dataset.sh
├── checkpoints
│   └── 2021-01-09
│       ├── osnet-ckpt-040_quant_edgetpu.log
│       ├── osnet-ckpt-040_quant_edgetpu.tflite
│       ├── osnet-ckpt-040_quant.tflite
│       └── osnet-ckpt-040.tflite
├── config
│   ├── config.py
│   ├── __init__.py
├── evaluation.py
├── logs
│   └── 2021-01-09
├── models
│   ├── __init__.py
│   ├── models.py
│   ├── osnet.py
├── prepare_data
│   ├── build_imagenet_data.py
│   ├── imagenet_2012_validation_synset_labels.txt
│   ├── imagenet_lsvrc_2015_synsets.txt
│   ├── imagenet_metadata.txt
│   ├── __init__.py
│   ├── preprocess_imagenet_validation_data.py
│   └── synset_words.txt
├── README.md
├── requirements.txt
├── run_train.sh
├── train.py
└── utils
    ├── dataset.py
    ├── image_preprocessing.py
```




