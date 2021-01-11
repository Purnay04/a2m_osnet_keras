ROOT_DIR="/media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012"
cd prepare_data

echo "Building Validaton...."
python preprocess_imagenet_validation_data.py \
          $ROOT_DIR/validation \
          imagenet_2012_validation_synset_labels.txt


echo "Building TFRecords ..."
python build_imagenet_data.py \
          --output_directory $ROOT_DIR/tfrecords \
          --train_directory $ROOT_DIR/train \
          --validation_directory $ROOT_DIR/validation