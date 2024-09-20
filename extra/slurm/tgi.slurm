#!/usr/bin/env bash
#SBATCH --job-name tgi-benchmark
#SBATCH --output /fsx/%u/logs/%x-%j.log
#SBATCH --gpus 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 11
#SBATCH --mem-per-cpu 20G
#SBATCH --time 1:50:00
#SBATCH --partition hopper-prod
#SBATCH --qos normal

if [ -z "$MODEL" ]; then
    echo "MODEL environment variable is not set"
    exit 1
fi

#model="meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "Starting TGI benchmark for $MODEL"
export RUST_BACKTRACE=full
export RUST_LOG=text_generation_inference_benchmark=info
export PORT=8090
# start TGI
srun -u \
     -n 1 \
     --mem-per-cpu 20G \
     --cpus-per-task 11 \
     --gpus=1 \
     --exact \
     --container-image='ghcr.io#huggingface/text-generation-inference' \
     --container-env=PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     /usr/local/bin/text-generation-launcher \
      --model-id $MODEL \
      --max-concurrent-requests 512 \
      --cuda-graphs="2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92"&

# wait until /health is available, die after 5 minutes
timeout 300 bash -c "while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)\" != \"200\" ]]; do sleep 1 && echo \"Waiting for TGI to start...\"; done" || exit 1
exit_code=$?

RESULTS_DIR="/fsx/$USER/benchmarks_results/tgi"
mkdir -p "${RESULTS_DIR}"

if [[ $exit_code != 124 ]]; then
    # run benchmark
    echo "Starting benchmark"
    srun -u \
         -n 1 \
         --mem-per-cpu 20G \
         --cpus-per-task 11 \
         --gpus 0 \
         --exact \
         --container-image='registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/text-generation-inference-benchmark:latest' \
         --container-mounts="${RESULTS_DIR}:/opt/text-generation-inference-benchmark/results" \
         --no-container-mount-home \
         text-generation-inference-benchmark \
             --tokenizer-name $MODEL \
             --max-vus 800 \
             --url "http://localhost:${PORT}" \
             --duration 30s \
             --warmup 20s \
             --num-rates 2 \
             --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --no-console
fi

# stop TGI
scancel --signal=TERM "$SLURM_JOB_ID.0"

echo "End of benchmark"