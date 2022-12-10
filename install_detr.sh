#!/usr/bin/env bash

# chmod +x coco.sh

git clone https://github.com/cyoukaikai/SDDETR.git

mv SDDETR/* . 
rm -rf SDDETR


mkdir DN-DETR
mv DN-DETR.zip DN-DETR
cd DN-DETR
unzip DN-DETR.zip 
mv DN-DETR/* .

chmod +x *.sh
./install_sd.sh

