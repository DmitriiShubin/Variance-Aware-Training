FROM nvidia/cuda:11.2.0-devel-ubuntu18.04
FROM python:3.7.9
COPY ./requirements.txt .
RUN pip install -r requirements.txt


## Add ssh
#RUN apt-get update -y
#RUN apt-get install -y openssh-server
#
#RUN mkdir /var/run/sshd
#RUN mkdir /root/.ssh
## SSH login fix. Otherwise user is kicked off after login
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile
#EXPOSE 22
#COPY current_key /root/.ssh/authorized_keys
#COPY git_key /root/.ssh/id_rsa
#RUN update-rc.d ssh defaults
#
#COPY ./docker-entrypoint.sh /




