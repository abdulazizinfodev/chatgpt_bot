#!/bin/bash

echo "Pulling..."

git pull

echo "Buliding application..."

docker-compose --env-file config/config.env up -d --build