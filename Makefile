build:
	cargo build --release --package text-generation-inference-benchmark --bin text-generation-inference-benchmark

run: build
	cargo run --package text-generation-inference-benchmark --bin text-generation-inference-benchmark -- $@

test:
	cargo test --package text-generation-inference-benchmark

lint:
	cargo +nightly fmt
	cargo clippy --package text-generation-inference-benchmark
