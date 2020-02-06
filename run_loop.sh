#!/bin/bash

run_dir=.
out_base=${run_dir}/models
data_dir=../../Kinect/data/normalized

GPUID=-1

function run_once() {
  out_dir=${out_base}/H${n_hidden}_Z${n_latent}_L${n_layers}_B${beta}

  n_enc_layers=${n_layers}
  n_enc_hidden=${n_hidden}
  n_dec_layers=${n_layers}
  n_dec_hidden=${n_hidden}

  if [ ! -d ${out_dir} ]; then
    mkdir ${out_dir}
  fi
  echo -n "running in ${out_dir} ... "
  python ${run_dir}/train.py -e ${epoch} -i ${save_int} -b ${batch_size} -n ${n_enc_hidden} -d ${n_dec_hidden} -z ${n_latent} -l ${n_enc_layers} -k ${n_dec_layers} -B ${beta} -o ${out_dir} ${data_dir}/*.json >& ${out_dir}/train.log
  echo "done."
}

epoch=1000
save_int=20000
batch_size=5

for n_layers in 1 2
do
    for n_hidden in 100 300 500
    do
        for n_latent in 2 3 5
        do
            for beta in 0.01 0.05 0.1 0.5
            do
                run_once
            done
        done
    done
done
