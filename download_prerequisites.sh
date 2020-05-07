#!/usr/bin/bash

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z1fbJrNnjD7VrubYBCtfweYNoMjH1uEW' -O feat_photo.npz
mv feat_photo.npz SketchTriplet/out_feat_flickr15k_1904041458/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PqzIO-OWTeEAl3Hs5tRavRs6-qZ8OmXb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PqzIO-OWTeEAl3Hs5tRavRs6-qZ8OmXb" -O resize_img.zip
rm /tmp/cookies.txt
unzip resize_img.zip
mv resize_img/* static/images/dataset/
rm -r resize_img