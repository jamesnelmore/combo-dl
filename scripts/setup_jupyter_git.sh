#!/bin/bash

pip install nb-clean

git config --local filter.nb-clean.clean "nb-clean clean"
git config --local filter.nb-clean.required true

echo "Jupyter notebook Git configuration complete!" 