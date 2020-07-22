#!/bin/bash

run_dir=.
out_base=${run_dir}/models/tmp
data_dir=../../Kinect/data/normalized


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

epoch=100
save_int=20000
batch_size=10
beta=0.05

n_hidden=200
n_latent=5
n_layers=1
run_once
