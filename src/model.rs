use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::{
    generation::LogitsProcessor, models::quantized_mistral::Model as QMistral,
    utils::apply_repeat_penalty as apply_rp,
};
use tokenizers::Tokenizer;

pub struct TextGeneration {
    model: QMistral,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: QMistral,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<Vec<String>> {
        let mut tokens = self.tokenize(prompt)?;

        let eos_token = self.get_eos_token()?;

        for _ in 0..sample_len {
            let context_size = std::cmp::min(sample_len, tokens.len() - 1);
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, start_pos)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            let logits = self.apply_repeat_penalty(&logits, &tokens)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token.into());

            if next_token as i64 == eos_token {
                break;
            }
        }

        Ok(tokens
            .iter()
            .map(|&id| {
                self.tokenizer
                    .id_to_token(id as u32)
                    .unwrap()
                    .replace(' ', " ")
                    .replace("<0x0A>", "\n")
            })
            .collect())
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<i64>> {
        self.tokenizer
            .encode(prompt, true)
            .map(|encoding| encoding.get_ids().iter().map(|&id| id as i64).collect())
            .map_err(E::msg)
    }

    fn get_eos_token(&self) -> Result<i64> {
        self.tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .map(|id| id as i64)
            .ok_or_else(|| E::msg("cannot find the </s> token"))
    }

    fn apply_repeat_penalty(&self, logits: &Tensor, tokens: &[i64]) -> Result<Tensor> {
        if self.repeat_penalty == 1.0 {
            Ok(logits.clone())
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            let tokens_u32: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
            apply_rp(logits, self.repeat_penalty, &tokens_u32[start_at..]).map_err(|err| err.into())
        }
    }
}
