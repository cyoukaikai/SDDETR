#!/usr/bin/env bash




wget http://datasets.blockelite.cn/20.COCO_2017/annotations_trainval2017.zip
wget http://datasets.blockelite.cn/20.COCO_2017/train2017.zip
wget http://datasets.blockelite.cn/20.COCO_2017/val2017.zip

mkdir coco
mv *2017*.zip coco
cd coco

unzip '*.zip'


mkdir DN-DETR
mkdir DN-DETR_old/
ln -s coco DN-DETR/coco
ln -s coco DN-DETR_old/coco
