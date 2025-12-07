#!/bin/bash
# Setup script for Modal sandbox

set -e

# Update package lists
apt-get update

# Installs necessary packages
apt-get install -y nodejs npm curl wget
