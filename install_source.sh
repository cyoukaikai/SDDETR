#!/usr/bin/env bash

# export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc
#echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR

mv pip.conf ~/.pip/pip.conf
mv condarc.sh ~/.condarc



chmod +x *.sh



