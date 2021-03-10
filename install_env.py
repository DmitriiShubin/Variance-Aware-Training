import os

# install nvidia-docker-compose
os.system("pip install nvidia-docker-compose")

print('Enter sudo password')
sudoPassword = input()

# build the image
command = 'docker build . --tag jet_training_uada_base'
print("echo '%s'|sudo -S %s" % (sudoPassword, command))
p = os.system("echo '%s'|sudo -S %s" % (sudoPassword, command))


# sudo docker-compose up -d
