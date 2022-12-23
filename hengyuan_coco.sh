#!/usr/bin/env bash



oss cp -r -f -public oss://COCO/2017/annotations_trainval2017.zip /hy-tmp
oss cp -r -f -public oss://COCO/2017/val2017.zip /hy-tmp
oss cp -r -f -public oss://COCO/2017/train2017.zip /hy-tmp

cd /hy-tmp
mkdir coco
mv *2017*.zip coco

cd coco
unzip '*.zip'
