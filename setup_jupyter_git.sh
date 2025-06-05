#!/bin/bash

# Install nb-clean
pip install nb-clean

# Configure Git to use nb-clean
git config --local filter.nb-clean.clean "nb-clean clean"
git config --local filter.nb-clean.required true

echo "Jupyter notebook Git configuration complete!" 