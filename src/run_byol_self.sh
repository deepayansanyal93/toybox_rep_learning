#!/bin/bash

epochs=15

python ssl_impl.py -d self -data toybox -f1 0.02 -f2 0.1 -lr 0.1 -lr_ft 0.2 -e1 $epochs -e2 5 -ssl byol -rep 1

python ssl_impl.py -d self -data toybox -f1 0.02 -f2 0.1 -lr 0.2 -lr_ft 0.2 -e1 $epochs -e2 5 -ssl byol -rep 1