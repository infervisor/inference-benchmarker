# TGI Benchmark: A High-Performance Tool for Text Generation Model Benchmarking

Benchmarking inference servers for text generation models presents unique challenges.
The performance of these models can vary greatly depending on factors like input prompts,
decoding strategies, hardware specifications, and server configurations.

**TGI Benchmark** is designed to streamline this process by providing a comprehensive benchmarking tool
that evaluates the real-world performance of text generation models and servers.
With **TGI Benchmark**, you can easily test your model's throughput and efficiency under various workloads,
identify performance bottlenecks, and optimize your deployment for production environments.

It can be used to benchmark any text generation server that exposes an OpenAI-compliant API.

## Features

* Broad Compatibility: Benchmarks any text generation server with an OpenAPI-compliant chat API.
* Automatic Sweep Mode: Detects maximum throughput and sweeps in-between.
* Open-Loop Benchmarking: Uses constant arrival rates to simulate real-world workloads.
* High-Performance: Built with Rust 🦀 for high-performance benchmarking.
* JSON Output: Delivers performance results in a structured, easy-to-analyze format.

![ui.png](assets/ui.png)

## Table of contents

<!-- TOC -->
* [TGI Benchmark: A High-Performance Tool for Text Generation Model Benchmarking](#tgi-benchmark-a-high-performance-tool-for-text-generation-model-benchmarking)
  * [Features](#features)
  * [Table of contents](#table-of-contents)
  * [Get started](#get-started)
    * [Run a benchmark](#run-a-benchmark)
      * [1. Start an inference server](#1-start-an-inference-server)
      * [2. Run a benchmark using Docker image](#2-run-a-benchmark-using-docker-image)
    * [Configure your benchmark](#configure-your-benchmark)
      * [Benchmark mode](#benchmark-mode)
      * [Dataset configuration](#dataset-configuration)
      * [Prompt configuration](#prompt-configuration)
    * [Decode options](#decode-options)
  * [Deploy on Kubernetes](#deploy-on-kubernetes)
  * [Deploy on Slurm](#deploy-on-slurm)
  * [Development](#development)
  * [Frequently Asked Questions](#frequently-asked-questions)
  * [TODO](#todo)
<!-- TOC -->

## Get started

### Run a benchmark

#### 1. Start an inference server

**TGI**

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=<your HF READ token>

docker run --gpus all --shm-size 1g -p 8080:80 -e "HF_TOKEN=$HF_TOKEN" \
    ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $MODEL
```

**vLLM**

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=<your HF READ token>
docker run --runtime nvidia --gpus all \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8080:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL
```

#### 2. Run a benchmark using Docker image

```shell
MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=<your HF READ token>
# we mount results to the current directory
$ docker run \
    --rm \
    -it \
    --net host \
    -v $(pwd):/opt/text-generation-inference-benchmark/results \
    -e "HF_TOKEN=$HF_TOKEN" \
    ghcr.io/huggingface/text-generation-inference-benchmark:latest \
    text-generation-inference-benchmark \
    --tokenizer-name "$MODEL" \
    --max-vus 800 \
    --url http://localhost:8080 \
    --warmup 20s \
    --num-rates 10 \
    --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
    --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" 
```

Results will be saved in JSON format in current directory.

### Configure your benchmark

#### Benchmark mode

In default mode, tool runs a `sweep` benchmark. It first runs a throughput test to find the maximum throughput, then
sweeps on QPS values up to the maximum throughput.

Available modes:

- `sweep`: runs a sweep benchmark
- `rate`: runs a benchmark at a fixed request rate
- `throughput`: runs a benchmark at a fixed throughput (constant VUs)

Example running a benchmark at a fixed request rates:

```shell 
MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=<your HF READ token>
$ docker run \
    --rm \
    -it \
    --net host \
    -v $(pwd):/opt/text-generation-inference-benchmark/results \
    -e "HF_TOKEN=$HF_TOKEN" \
    ghcr.io/huggingface/text-generation-inference-benchmark:latest \
    text-generation-inference-benchmark \
    --tokenizer-name "meta-llama/Llama-3.1-8B-Instruct" \
    --max-vus 800 \
    --duration 120s \
    --url http://localhost:8080 \
    --warmup 30s \
    --benchmark-kind rate \
    --rates 1.0 \
    --rates 5.0 \
    --rates 10.0 \
    --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
    --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10"
```

#### Dataset configuration

Prompts are sampled for a Hugging Face dataset file, using a [subset of ShareGPT
as default](https://huggingface.co/datasets/hlarcher/share_gpt_small). You can specify a different dataset file using
the
`--dataset` and `--dataset-file` option.

Dataset is expected to be JSON with the following format:

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "rewrite that entire paragraph in the same style like this one: "
      }
    ]
  }
]
```

To benchmark with prefix caching, you can use a system prompt that will be sent with each request from a discussion.

```json
[
  {
    "conversations": [
      {
        "role": "system",
        "content": "You are a helpful assistant that makes jokes at each response."
      },
      {
        "role": "user",
        "content": "rewrite that entire paragraph in the same style like this one:"
      }
    ]
  }
]
```

#### Prompt configuration

For consistent results you can configure the token count and variance. The tool will sample prompts with the specified
values, sampling token counts from a normal distribution with the specified variance.

```shell
--prompt-options "num_tokens=50,max_tokens=60,min_tokens=40,variance=10"
```

### Decode options

You can also configure the decoding options for the model. The tool will sample decoding options with the specified
values, sampling token counts from a normal distribution with the specified variance.

```shell
--decode-options "num_tokens=50,max_tokens=60,min_tokens=40,variance=10"
```

## Deploy on Kubernetes

You can deploy the benchmarking tool on Kubernetes using the provided Helm chart.

Review the values (especially model, HF token and resources), and install the chart:
```shell
$ helm install tgi-benchmark ./extra/k8s/text-generation-inference-benchmark
```

## Deploy on Slurm

Slurm example is provided in `extra/slurm`.

## Development

You need [Rust](https://rustup.rs/) installed to build the benchmarking tool.

```shell
$ make build
```

## Frequently Asked Questions

* **What's the difference between constant arrival rate and constant virtual user count?**
    * **Constant virtual user count** means that the number of virtual users is fixed. Each virtual user can send a
      single requests and waits for server response. It's basically simulating a fixed number of users querying the
      server.
    * **Constant arrival rate** means that the rate of requests is fixed and the number of virtual users is adjusted to
      maintain that rate. Queries hit the server independently of responses performances.

  **Constant virtual user count** is a closed loop model where the server's response time dictates the number of
  iterations. **Constant arrival rate** is an open-loop model more representative of real-life workloads.


* **Why do I get high error rate when running `thoughput` benchmark?**

  Throughput bench tries to saturate the server with a high request rate. The error rate is high because the server is
  not able to handle the request rate or rate limiting the requests.
  In the case of TGI, this is controlled by the `--max-concurrent-requests` option.


* **What is the influence of CUDA graphs?**

  CUDA graphs are used to optimize the GPU usage by minimizing the overhead of launching kernels. This can lead to
  better performance in some cases, but can also lead to worse performance in others.
  If your CUDA graphs are not evenly distributed, you may see a performance drop at some request rates as batch size may
  fall in a bigger CUDA graph batch size leading to a lost of compute due to excessive padding.

* **I get less tokens generated than expected in a benchmark.**

  Inference servers use `max_tokens` parameter to limit the number of tokens generated. If the model
  generates an end-of-sequence token before reaching `max_tokens`, the generation will stop.
  There is currently no way to guarantee a fixed number of tokens generated without modifying the inference server.
  So you may have `(successful requests) * max_tokens < generated tokens`.

## TODO

- [X] Customizable token count and variance
- [X] Check results
- [X] Allow for system prompts for prefix caching
- [ ] Allow for multi-turn prompts
- [X] Script to generate plots from results
- [X] Add support for multiple tokens in stream chunks (when speculation is active)
