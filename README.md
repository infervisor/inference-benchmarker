# Text Generation Inference benchmarking tool

A lightweight benchmarking tool for inference servers.
Benchmarks using constant arrival rate or constant virtual user count.

![ui.png](assets%2Fui.png)

## TODO
- [ ] Customizable token count and variance
- [ ] Check results
- [ ] Allow for multiturn prompts for speculation
- [ ] Push results to Optimum benchmark backend
- [ ] Script to generate plots from results

## Running a benchmark

```
# start a TGI/vLLM server somewhere, then run benchmark...
# ... we mount results to the current directory
$ docker run \
    --rm \
    -it \
    --net host \
    -v $(pwd):/opt/text-generation-inference-benchmark/results \
    registry.internal.huggingface.tech/api-inference/text-generation-inference-benchmark:latest \
    text-generation-inference-benchmark \
    --tokenizer-name "Qwen/Qwen2-7B" \
    --max-vus 800 \
    --url http:/localhost:8080 \
    --warmup 20s
```

Results will be saved in `results.json` in current directory.