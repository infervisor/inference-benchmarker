#!/usr/bin/env bash
#SBATCH --job-name tgi-benchmark
#SBATCH --output /fsx/%u/logs/%x-%j.log
#SBATCH --time 1:50:00
#SBATCH --qos normal
#SBATCH --partition hopper-prod
#SBATCH --gpus 1 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1
#SBATCH hetjob
#SBATCH --gpus 1 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1


if [ -z "$MODEL" ]; then
    echo "MODEL environment variable is not set"
    exit 1
fi

echo "Starting TGI benchmark for $MODEL"
export RUST_BACKTRACE=full
export RUST_LOG=text_generation_inference_benchmark=info
export PORT=8090

echo "Model will run on ${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}"
echo "Benchmark will run on ${SLURM_JOB_NODELIST_HET_GROUP_1}"

# start TGI
srun --het-group=0 \
     -u \
     -n 1 \
     --container-image='ghcr.io#huggingface/text-generation-inference' \
     --container-env=PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     /usr/local/bin/text-generation-launcher \
      --model-id $MODEL \
      --max-concurrent-requests 512 \
      --cuda-graphs="1,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128"&

# wait until /health is available, die after 5 minutes
timeout 300 bash -c "while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)\" != \"200\" ]]; do sleep 1 && echo \"Waiting for TGI to start...\"; done" || exit 1
exit_code=$?

RESULTS_DIR="/fsx/$USER/benchmarks_results/tgi"
mkdir -p "${RESULTS_DIR}"

if [[ $exit_code != 124 ]]; then
    # run benchmark
    echo "Starting benchmark"
    srun --het-group=1 \
         -u \
         -n 1 \
         --container-image="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech#library/text-generation-inference-benchmark:latest" \
         --container-mounts="${RESULTS_DIR}:/opt/text-generation-inference-benchmark/results" \
         --no-container-mount-home \
         text-generation-inference-benchmark \
             --tokenizer-name "$MODEL" \
             --max-vus 800 \
             --url "http://${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}" \
             --duration 30s \
             --warmup 30s \
             --num-rates 2 \
             --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --no-console
fi

# stop TGI
scancel --signal=TERM "$SLURM_JOB_ID+0"

echo "End of benchmark"