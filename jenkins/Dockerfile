FROM jenkins/jenkins:lts
USER root
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
# Installer Docker dans une image Docker.
RUN curl -sSL https://get.docker.com/ | sh 
