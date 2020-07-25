#!/usr/bin/env bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

echo "Downloading dataset..."
gdrive_download 1PEukDlZ6IkEhh9XfTTMMtFOwdXOC3iKn fonts_meta.csv
gdrive_download 15xPf2FrXaHZ0bf6htZzc9ORTMGHYz9DX fonts_tensor.zip

echo "Download done. Unzipping..."
unzip fonts_tensor.zip
echo "Done."
