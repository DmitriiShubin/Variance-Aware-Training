chmod 600 ~/.ssh/id_rsa
service ssh start
ssh-agent >> ~/.bashrc
echo "ssh-add ~/.ssh/id_rsa" >> ~/.bashrc
export SHELL=/bin/bash