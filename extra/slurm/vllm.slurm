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

echo "Starting vLLM benchmark for $MODEL"
export RUST_BACKTRACE=full
export RUST_LOG=text_generation_inference_benchmark=info;
export PORT=8090
# start TGI
srun -u \
     -n 1 \
     --mem-per-cpu 20G \
     --cpus-per-task 11 \
     --gpus=1 \
     --exact \
     --container-image='vllm/vllm-openai:latest' \
     --container-env=PORT \
     --container-mounts="/scratch:/root/.cache/huggingface" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     python3 -m vllm.entrypoints.openai.api_server --model "${MODEL}" --port "${PORT}" &

# wait until /health is available, die after 5 minutes
timeout 300 bash -c "while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)\" != \"200\" ]]; do sleep 1 && echo \"Waiting for vLLM to start...\"; done" || exit 1
exit_code=$?

RESULTS_DIR="/fsx/$USER/benchmarks_results/vllm"
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
             --tokenizer-name "$MODEL" \
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