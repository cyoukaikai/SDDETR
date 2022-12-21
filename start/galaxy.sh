


<<COMMENT
echo g2J3C7FN
sftp -P 15022 12318626145@pan.blockelite.cn

echo "get cudnn-11.3-linux-x64-v8.2.0.53.tgz"
echo "get cuda_11.3.0_465.19.01_linux.run"
echo "get SDDETR.zip"


sudo sh cuda_11.3.0_465.19.01_linux.run
sudo apt-get install nano
nano ~/.bashrc


COMMENT

sudo apt-get install nano
sudo apt-get install tmux



unzip SDDETR.zip
mv SDDETR/* .
chmod +x *.sh
chmod +x DN-DETR/*.sh


mv pip.conf ~/.pip/pip.conf
mv condarc_beiwai.sh ~/.condarc

###############################
# https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local
###############################
#wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run



#Downloads/cudnn-11.3-linux-x64-v8.2.0.53.tgz


#sudo sh cuda_11.3.0_465.19.01_linux.run

tar -xvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*


mv bashrc ~/.bashrc
source ~/.bashrc
echo $CUDA_HOME
readlink -f /usr/local/cuda
# /usr/local/cuda-11.3



conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch



./install_sd.sh

conda install matplotlib

echo 8b14dd204de425f8a0c700ab58d00ec9ce60db4b
# tmux new -s train
wandb login

<<COMMENT
echo g2J3C7FN
sftp -P 15022 12318626145@pan.blockelite.cn

echo "get cudnn-11.3-linux-x64-v8.2.0.53.tgz"
echo "get cuda_11.3.0_465.19.01_linux.run"


# uncheck install driver 

mv * ../
cd ..
mv galaxy/* ~/.
COMMENT


