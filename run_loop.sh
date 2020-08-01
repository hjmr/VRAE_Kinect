#!/bin/bash

run_dir=.
out_base=${run_dir}/models
data_dir=../../Kinect/data/normalized

function run_once() {
  out_dir=${out_base}/H${n_hidden}_Z${n_latent}_L${n_layers}_B${beta}_BS${batch_size}_${loop_count}

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

for batch_size in 10
do
    for n_layers in 1 2
    do
        for n_hidden in 100 200
        do
            for n_latent in 10 20
            do
                for beta in 1.0
                do
                    for loop_count in 01 02 03 04 05 06 07 08 09 10
                    do
                        run_once
                    done
                done
            done
        done
    done
done
