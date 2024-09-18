build:
	cargo build --release --package text-generation-inference-benchmark --bin text-generation-inference-benchmark

run: build
	cargo run --package text-generation-inference-benchmark --bin text-generation-inference-benchmark -- $@