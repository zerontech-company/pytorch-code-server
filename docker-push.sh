#!/bin/sh
docker build -t localhost:5001/zaant-codeserver .
docker image tag localhost:5001/zaant-codeserver zerontech/pytorch-1.8.1-cuda11.1-cudnn8-devel-codeserver
docker image push zerontech/pytorch-1.8.1-cuda11.1-cudnn8-devel-codeserver