#!/bin/sh
docker build -t localhost:5001/zaant-codeserver .
docker image push localhost:5001/zaant-codeserver