#!/usr/bin/env bash
pip install --upgrade pip
pip install -r requirements.txt
cd src
git clone https://github.com/nschejtman/caffe-posenet.git

export PYTHONPATH=$(pwd)