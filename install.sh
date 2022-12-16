#!/usr/bin/env bash



git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR
chmod +x *.sh

# mv pip.conf 


./install_sd.sh


echo 8b14dd204de425f8a0c700ab58d00ec9ce60db4b
# tmux new -s train
wandb login
