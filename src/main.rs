use anyhow::{Error, Result};
use candle_core::Device;
use candle_transformers::models::{mistral::Config, quantized_mistral::Model as QMistral};

use std::path::PathBuf;
use tokenizers::Tokenizer;

mod model;
use model::TextGeneration;

fn load_model_and_tokenizer(
    tokenizer_path: PathBuf,
    model_path: PathBuf,
    device: &Device,
) -> Result<(Tokenizer, QMistral, Config)> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
    let config = Config::config_7b_v0_1(true);
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_path, &device)
        .map_err(|err| Error::from(err))?;
    let model = QMistral::new(&config, vb)?;

    Ok((tokenizer, model, config))
}

fn main() -> Result<()> {
    let tokenizer_filename = PathBuf::from("./models/tokenizer.json");
    let model_name = PathBuf::from("./models/model-q4k.gguf");

    let device = Device::Cpu;
    let (tokenizer, model, _config) =
        load_model_and_tokenizer(tokenizer_filename, model_name, &device)?;

    let seed = 299792458;
    let temperature = Some(0.8);
    let top_p = Some(0.9);
    let repeat_penalty = 1.1;
    let repeat_last_n = 64;
    let sample_len = 400;

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &device,
    );
    let generated_text = pipeline.run("...".into(), sample_len)?;
    println!("Generated Text: {}", generated_text.join(""));

    Ok(())
}
