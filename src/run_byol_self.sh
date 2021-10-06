#!/bin/bash

epochs=20
frac=0.02
b_size=64
workers=8

set -x

python ssl_impl.py -d self -data toybox -f1 $frac -f2 0.1 -lr 0.2 -lr_ft 0.2 -b $b_size -w $workers -e1 $epochs -e2 25 -ht -ssl byol -rep 1 -sv -sn byol_self_ht_1 -sr 20 -ef 20

python ssl_impl.py -d self -data toybox -f1 $frac -f2 0.1 -lr 0.5 -lr_ft 0.2 -b $b_size -w $workers -e1 $epochs -e2 25 -ht -ssl byol -rep 1 -sv -sn byol_self_ht_2 -sr 20 -ef 20

python ssl_impl.py -d self -data toybox -f1 $frac -f2 0.1 -lr 0.8 -lr_ft 0.2 -b $b_size -w $workers -e1 $epochs -e2 25 -ht -ssl byol -rep 1 -sv -sn byol_self_ht_3 -sr 20 -ef 20