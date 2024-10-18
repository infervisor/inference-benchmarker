FROM rust:1-bullseye AS builder
LABEL org.opencontainers.image.source=https://github.com/huggingface/text-generation-inference-benchmark
LABEL org.opencontainers.image.description="A benchmark tool for LLM inference engines"
LABEL org.opencontainers.image.licenses="Apache-2.0"
ARG GIT_SHA
WORKDIR /usr/src/text-generation-inference-benchmark
COPY . .
RUN cargo install --path .
FROM debian:bullseye-slim
RUN mkdir -p /opt/text-generation-inference-benchmark/results
WORKDIR /opt/text-generation-inference-benchmark
COPY --from=builder /usr/local/cargo/bin/text-generation-inference-benchmark /usr/local/bin/text-generation-inference-benchmark
CMD ["text-generation-inference-benchmark"]