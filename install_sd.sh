#!/usr/bin/env bash


install_detr() {
	dir_=$1


	cur_path_=$PWD
	cd $dir_
	
	chmod +x *.sh
	
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
	pip install opencv-python-headless==4.5.2.52
	
	touch ~/.tmux.conf
	echo "setw -g mouse on" >> ~/.tmux.conf
	tmux source-file ~/.tmux.conf

}


############################################

install_gcp_coco() {
	chmod +x coco.sh
	./coco.sh
}

# git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR
unzip 'DN-DETR*.zip' 

#mv DN-DETR/DN-DETR/* DN-DETR/
#rm -rf DN-DETR
install_detr DN-DETR

#mv DN-DETR_old/DN-DETR_old/* DN-DETR_old/ 
#install_detr DN-DETR_old

###################
install_other

# gdown --folder https://drive.google.com/drive/folders/1h4PB9DO7_I0TRwU62Wy0x3ixti8Gfjc3



