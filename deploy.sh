#!/bin/bash

sudo apt-get update
sudo apt-get install docker.io docker-compose

echo "Pulling..."

git pull

echo "Buliding application..."

docker-compose --env-file config/config.env up -d --build