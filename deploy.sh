#!/bin/bash

echo "Pulling..."

git pull

echo "Building..."

docker-compose --env-file config/config.env up --build