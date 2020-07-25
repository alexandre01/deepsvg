#!/usr/bin/env bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

echo "Downloading pretrained models..."
gdrive_download 1tsVx_cnFunSf5vvPWPVTjZ84IQC2pIDm hierarchical_ordered.pth.tar
gdrive_download 11KBUWfexw3LDvSFOVxy072_VCFYKm3L- hierarchical_ordered_fonts.pth.tar
echo "Done."
