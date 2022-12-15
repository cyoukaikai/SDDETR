#!/usr/bin/env bash


install_detr() {
	dir_=$1


	cur_path_=$PWD
	cd $dir_

	cd smrc
	pip install -r requirements.txt

	cd ../
	pip install -r requirements.txt


	pip install einops
	pip install -U openmim
	mim install mmcv-full
	pip install mmdet

	cd models/dn_dab_deformable_detr/ops
	python setup.py build install
	# python test.py
	cd ../../..

	cd $cur_path_
}


install_other() {
	pip install wandb

	touch ~/.tmux.conf
	echo "setw -g mouse on" >> ~/.tmux.conf
	tmux source-file ~/.tmux.conf

}

install_zhinengyun_coco() {
 wget http://datasets.blockelite.cn/20.COCO_2017/train2017.zip
 
}

############################################

install_gcp_coco() {
	chmod +x coco.sh
	./coco.sh
}

git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR
unzip '*.zip' #unzip DN-DETR_old.zip 
mv DN-DETR/DN-DETR/* DN-DETR/
mv -rf DN-DETR

install_detr DN-DETR

mv DN-DETR_old/DN-DETR_old/* DN-DETR_old/ 
mv -rf DN-DETR_old
install_detr DN-DETR_old

###################
install_other


echo 8b14dd204de425f8a0c700ab58d00ec9ce60db4b
# tmux new -s train
wandb login
