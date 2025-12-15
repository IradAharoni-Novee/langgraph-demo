#!/bin/bash
# Setup script for sandbox environments

set -e

# Use sudo only if not running as root
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Update package lists
$SUDO apt-get update

# Installs necessary packages
$SUDO apt-get install -y nodejs npm curl wget
