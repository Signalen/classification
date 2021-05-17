#!/usr/bin/env bash

set -u   # crash on missing env variables
set -e   # stop on any error
set -x   # print what we are doing

curl -d '{"text":"poep op straat"}' -H "Content-Type: application/json" -X POST http://localhost:8140/signals_mltool/predict
