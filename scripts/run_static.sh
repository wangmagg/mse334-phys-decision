#!/bin/bash

echo "phys better"
python3 -m src.main \
    --n-sample-iter 200 \
    --n-z-iter 10 \
    --adapt-type static \
    --p-phys-pos 0.1 \
    --p-phys-null 0.2 \
    --p-phys-neg 0.9 \
    --p-mdl-pos 0.4 \
    --p-mdl-null 0.2 \
    --p-mdl-neg 0.6 \
    --alpha0 0.01 \
    --alpha1 100 

echo "mdl better"
python3 -m src.main \
    --n-sample-iter 200 \
    --n-z-iter 10 \
    --adapt-type static \
    --p-phys-pos 0.4 \
    --p-phys-null 0.2 \
    --p-phys-neg 0.6 \
    --p-mdl-pos 0.1 \
    --p-mdl-null 0.2 \
    --p-mdl-neg 0.9 \
    --alpha0 0.01 \
    --alpha1 100 
done
