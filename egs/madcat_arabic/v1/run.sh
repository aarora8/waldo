#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. ./scripts/parse_options.sh  # e.g. this parses the --stage option if supplied.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

nj=70

download_dir1=/Users/ashisharora/google_Drive/madcat_arabic/LDC2012T15
download_dir2=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T09
download_dir3=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T15
writing_condition1=/Users/ashisharora/google_Drive/madcat_arabic/LDC2012T15/writing_conditions.tab
writing_condition2=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T09/writing_conditions.tab
writing_condition3=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T15/writing_conditions.tab
data_splits=/Users/ashisharora/google_Drive/madcat_arabic/madcat.dev.raw.lineid
out_dir=/Users/ashisharora/google_Drive/madcat_arabic/masks

#local/check_dependencies.sh


if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
      --download_dir3 $download_dir3 --writing_condition1 $writing_condition1 \
      --writing_condition2 $writing_condition2 --writing_condition3 $writing_condition3 \
      --data_splits $data_splits --out_dir $out_dir
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
#    --train-dir data \
#    --train-image-size 128 \
#    --core-config $dir/configs/core.config \
#    --unet-config $dir/configs/unet.config \
#    $dir/model_best.pth.tar
#
#fi

