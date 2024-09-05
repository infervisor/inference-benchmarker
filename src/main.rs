use text_generation_inference_benchmark::run;

#[tokio::main]
async fn main() {
    env_logger::init();
    println!("Hello, world!");
    run().await;
}
