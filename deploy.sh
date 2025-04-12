#!/bin/bash

# Stop and remove existing container
docker stop qr-mosaic-app || true
docker rm qr-mosaic-app || true

# Remove existing image
docker rmi qr-mosaic || true

# Build new image
docker build -t qr-mosaic .

# Run new container
docker run -d -p 80:8000 --name qr-mosaic-app qr-mosaic 