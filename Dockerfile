FROM rust:1-bullseye AS builder
WORKDIR /usr/src/text-generation-inference-benchmark
COPY . .
RUN cargo install --path .
FROM debian:bullseye-slim
RUN mkdir -p /opt/text-generation-inference-benchmark/results
WORKDIR /opt/text-generation-inference-benchmark
COPY --from=builder /usr/local/cargo/bin/text-generation-inference-benchmark /usr/local/bin/text-generation-inference-benchmark
CMD ["text-generation-inference-benchmark"]