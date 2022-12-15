#!/usr/bin/env bash

# chmod +x coco.sh

git clone https://github.com/cyoukaikai/SDDETR.git

mv SDDETR DN-DETR
mv DN-DETR_old.zip DN-DETR
cd DN-DETR
unzip DN-DETR_old.zip 
mv DN-DETR/* .
mv DN-DETR_old/* .
cd ../

chmod +x *.sh
./install_sd.sh
