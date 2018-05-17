#!/usr/bin/env bash

set -e
stage=0
nj=3
download_dir1=/Users/ashisharora/google_Drive/madcat_arabic/LDC2012T15
download_dir2=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T09
download_dir3=/Users/ashisharora/google_Drive/madcat_arabic/LDC2013T15
data=/Users/ashisharora/google_Drive/madcat_arabic/masks
mkdir -p data/dev/masks
. ./cmd.sh
. ./path.sh
. parse_options.sh
./local/check_tools.sh

echo "Date: $(date)."
if [ $stage -le 0 ]; then
  for dataset in dev; do
  dataset_file=/Users/ashisharora/google_Drive/madcat_arabic/madcat.dev.raw.lineid
  local/generate_mask_from_page_image.py $download_dir1 $download_dir2 $download_dir3 $dataset_file $data
  done
fi
echo "Date: $(date)."