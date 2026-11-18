//! Dump token ids for a text using a GGUF's embedded tokenizer (no weights needed).
//!
//! Usage: cargo run -p artificer-core --example tokenize -- MODEL.gguf (-f FILE | TEXT) [--no-bos]
//! Prints `count=N` then one `id<TAB>piece` line per token. Used to compare against
//! Ollama / llama.cpp tokenization of the same GGUF and text.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: tokenize MODEL.gguf (-f FILE | TEXT) [--no-bos]");
        return ExitCode::from(2);
    }
    let model_path = &args[0];
    let (text, rest) = if args[1] == "-f" {
        let Some(path) = args.get(2) else {
            eprintln!("-f needs a file");
            return ExitCode::from(2);
        };
        match std::fs::read_to_string(path) {
            Ok(t) => (t, &args[3..]),
            Err(e) => {
                eprintln!("read {path}: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        (args[1].clone(), &args[2..])
    };
    let add_special = !rest.iter().any(|a| a == "--no-bos");
    let model = match gguf::GgufModel::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load {model_path}: {e}");
            return ExitCode::from(1);
        }
    };
    let tok = match artificer_core::Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::from(1);
        }
    };
    let ids = match tok.encode(&text, add_special) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("encode: {e}");
            return ExitCode::from(1);
        }
    };
    println!("count={}", ids.len());
    for id in &ids {
        println!("{id}\t{:?}", tok.id_to_token(*id).unwrap_or("?"));
    }
    ExitCode::SUCCESS
}
