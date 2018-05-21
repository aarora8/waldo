#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. parse_options.sh  # e.g. this parses the --stage option if supplied.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

nj=70
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data

local/check_dependencies.sh


if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
fi


#epochs=10
#depth=5
#dir=exp/unet_${depth}_${epochs}_sgd
#if [ $stage -le 1 ]; then
#  # training
#  local/run_unet.sh --epochs $epochs --depth $depth
#fi
#
#if [ $stage -le 2 ]; then
#    echo "doing segmentation...."
#  local/segment.py \
#    --dir $dir \
#    --train-dir data/train_val \
#    --train-image-size 128 \
#    --core-config $dir/configs/core.config \
#    --unet-config $dir/configs/unet.config \
#    $dir/model_best.pth.tar
#
#fi

