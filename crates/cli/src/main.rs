//! Artificer CLI — local LLM inference engine.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use artificer_core::{
    avx2_enabled, avx512_enabled, avx512_vnni_enabled, current_rss_bytes, peak_rss_bytes,
    pool_workers, profile_dump, profile_env_enabled, profile_reset, render_user_chat_prompt,
    RenderedPrompt, SystemMemory,
};
use artificer_runtime::{
    cuda_feature_enabled, resolve_backend_kind, BackendKind, Engine, EngineLoadOptions,
    GenerateConfig,
};
use artificer_server::{serve_blocking, serve_mcp_stdio_blocking, ServerConfig, ServerError};
use clap::{Parser, Subcommand};
use gguf::{GgufModel, LoadMode, LoadOptions};

#[derive(Parser, Debug)]
#[command(
    name = "artificer",
    version,
    about = "Artificer — local LLM inference engine (GGUF / CPU-first)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Load a GGUF file and print metadata + tensor inventory.
    Inspect {
        model: PathBuf,
        #[arg(long)]
        metadata: bool,
        #[arg(long)]
        tensors: bool,
        #[arg(long)]
        no_mmap: bool,
    },

    /// Run one-shot text generation.
    Run {
        model: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value_t = artificer_core::DEFAULT_MAX_TOKENS as usize)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 40)]
        top_k: i32,
        #[arg(long)]
        seed: Option<u64>,
        /// Stop generation when this substring appears (repeatable).
        #[arg(long = "stop")]
        stop: Vec<String>,
        /// Pass prompt through raw (skip chat template formatting).
        #[arg(long)]
        raw: bool,
        /// Max context length (default 8192; also hard-cap for auto RAM budget).
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Disable automatic context shrink to fit RAM.
        #[arg(long)]
        no_auto_ctx: bool,
        /// Store KV cache in f16 (~half RAM).
        #[arg(long)]
        kv_f16: bool,
        /// Compute backend: `cpu` (default) or `cuda` (needs `--features cuda`).
        /// Overrides `ARTIFICER_BACKEND` when set.
        #[arg(long, value_name = "cpu|cuda")]
        backend: Option<String>,
        #[arg(long)]
        no_mmap: bool,
        /// Allow MoE models whose family is still experimental (prints a one-line warning).
        /// Without it they are refused. CUDA refuses MoE either way.
        #[arg(long)]
        experimental_moe: bool,
    },

    /// Benchmark prefill + decode tok/s and peak RSS.
    Bench {
        model: PathBuf,
        /// Prompt used for the timed run.
        #[arg(short, long, default_value = "What color is the sky? Explain briefly.")]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long)]
        no_auto_ctx: bool,
        #[arg(long)]
        kv_f16: bool,
        /// Compute backend: `cpu` (default) or `cuda` (needs `--features cuda`).
        /// Overrides `ARTIFICER_BACKEND` when set.
        #[arg(long, value_name = "cpu|cuda")]
        backend: Option<String>,
        /// Number of timed generate passes after the warm-up(s).
        #[arg(long, default_value_t = 1)]
        runs: u32,
        /// Untimed warm-up generates in the same process before the timed runs
        /// (first-call setup: cuBLAS handle, f16 tile scratch, …). 0 = time a cold first call.
        #[arg(long, default_value_t = 1)]
        warmup: u32,
        /// Allow MoE models whose family is still experimental (prints a one-line warning).
        /// Without it they are refused. CUDA refuses MoE either way.
        #[arg(long)]
        experimental_moe: bool,
    },

    /// Start the OpenAI-compatible HTTP server (+ MCP at POST /mcp).
    Serve {
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[command(flatten)]
        security: ServeSecurityArgs,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Model id returned by `/v1/models` and completions.
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long)]
        no_auto_ctx: bool,
        #[arg(long)]
        kv_f16: bool,
        /// Compute backend: `cpu` (default) or `cuda` (needs `--features cuda`).
        /// Overrides `ARTIFICER_BACKEND` when set.
        #[arg(long, value_name = "cpu|cuda")]
        backend: Option<String>,
        /// If `--backend cuda` cannot start (no device, no cuBLAS, VRAM, or a
        /// non-CUDA build), serve on CPU and report `fallback_reason` in
        /// `GET /health` instead of exiting. Also `ARTIFICER_CPU_FALLBACK=1`.
        #[arg(long)]
        cpu_fallback: bool,
        /// Allow MoE models whose family is still experimental (prints a one-line warning).
        /// Without it they are refused. CUDA refuses MoE either way.
        #[arg(long)]
        experimental_moe: bool,
    },

    /// Run as an MCP server over stdio (JSON-RPC, newline-delimited).
    ///
    /// Tools: `chat`, `complete`. Logs go to stderr; protocol on stdin/stdout.
    Mcp {
        model: PathBuf,
        /// Model id advertised in serverInfo / tooling context.
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long)]
        no_auto_ctx: bool,
        #[arg(long)]
        kv_f16: bool,
        /// Compute backend: `cpu` (default) or `cuda` (needs `--features cuda`).
        /// Overrides `ARTIFICER_BACKEND` when set.
        #[arg(long, value_name = "cpu|cuda")]
        backend: Option<String>,
        /// Allow MoE models whose family is still experimental (prints a one-line warning).
        /// Without it they are refused. CUDA refuses MoE either way.
        #[arg(long)]
        experimental_moe: bool,
    },
    /// Perplexity over a text file, matching llama.cpp `llama-perplexity` (default mode at
    /// 8642541b): whole-file tokenization, `-c` chunks with BOS replacing each chunk's first
    /// token, second half of each chunk scored. Prints `[k]ppl,` per chunk and
    /// `Final estimate: PPL = X +/- Y`.
    Perplexity {
        model: PathBuf,
        /// Text file to evaluate (one trailing newline is dropped, as llama.cpp `-f` does).
        #[arg(short = 'f', long = "file")]
        file: PathBuf,
        /// Chunk / context length (llama.cpp `-c`).
        #[arg(short = 'c', long = "ctx", default_value_t = 512)]
        ctx: u32,
        /// Evaluate at most this many chunks (default: all whole chunks).
        #[arg(long)]
        chunks: Option<usize>,
        /// Matmul worker threads (sets `ARTIFICER_POOL`; default: available cores, max 16).
        #[arg(short = 't', long)]
        threads: Option<usize>,
        /// Store KV cache in f16 (default f32).
        #[arg(long)]
        kv_f16: bool,
        /// Compute backend: `cpu` (default) or `cuda` (needs `--features cuda`).
        #[arg(long, value_name = "cpu|cuda")]
        backend: Option<String>,
        /// Debug: write the file's token ids (one per line) here.
        #[arg(long)]
        dump_tokens: Option<PathBuf>,
        /// Allow MoE models whose family is still experimental (prints a one-line warning).
        /// Without it they are refused. CUDA refuses MoE either way.
        #[arg(long)]
        experimental_moe: bool,
    },
}

fn main() -> ExitCode {
    // Always log to stderr so stdout stays free for MCP stdio JSON-RPC.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect {
            model,
            metadata,
            tensors,
            no_mmap,
        } => inspect(model, metadata, tensors, no_mmap),
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            stop,
            raw,
            ctx,
            no_auto_ctx,
            kv_f16,
            backend,
            no_mmap: _,
            experimental_moe,
        } => cmd_run(
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            stop,
            raw,
            load_opts(
                ctx,
                no_auto_ctx,
                kv_f16,
                backend.as_deref(),
                false,
                experimental_moe,
            )?,
        ),
        Commands::Bench {
            model,
            prompt,
            max_tokens,
            ctx,
            no_auto_ctx,
            kv_f16,
            backend,
            runs,
            warmup,
            experimental_moe,
        } => cmd_bench(
            model,
            prompt,
            max_tokens,
            load_opts(
                ctx,
                no_auto_ctx,
                kv_f16,
                backend.as_deref(),
                false,
                experimental_moe,
            )?,
            runs,
            warmup,
        ),
        Commands::Serve {
            model,
            host,
            security,
            port,
            model_id,
            ctx,
            no_auto_ctx,
            kv_f16,
            backend,
            cpu_fallback,
            experimental_moe,
        } => cmd_serve_with_security(
            model,
            host,
            port,
            model_id,
            load_opts(
                ctx,
                no_auto_ctx,
                kv_f16,
                backend.as_deref(),
                cpu_fallback || env_cpu_fallback(),
                experimental_moe,
            )?,
            security,
        ),
        Commands::Mcp {
            model,
            model_id,
            ctx,
            no_auto_ctx,
            kv_f16,
            backend,
            experimental_moe,
        } => cmd_mcp(
            model,
            model_id,
            load_opts(
                ctx,
                no_auto_ctx,
                kv_f16,
                backend.as_deref(),
                false,
                experimental_moe,
            )?,
        ),
        Commands::Perplexity {
            model,
            file,
            ctx,
            chunks,
            threads,
            kv_f16,
            backend,
            dump_tokens,
            experimental_moe,
        } => cmd_perplexity(
            model,
            file,
            ctx,
            chunks,
            threads,
            kv_f16,
            backend.as_deref(),
            dump_tokens,
            experimental_moe,
        ),
    }
}

fn load_opts(
    ctx: u32,
    no_auto_ctx: bool,
    kv_f16: bool,
    backend_flag: Option<&str>,
    cpu_fallback: bool,
    experimental_moe: bool,
) -> Result<EngineLoadOptions> {
    let backend = resolve_backend_kind(backend_flag)?;
    if backend == BackendKind::Cuda && !cuda_feature_enabled() && !cpu_fallback {
        anyhow::bail!(
            "CUDA backend requested but this binary was built without `--features cuda`; rebuild with `cargo build -p artificer-cli --release --features cuda` or use `--backend cpu`"
        );
    }
    Ok(EngineLoadOptions {
        auto_context: !no_auto_ctx,
        max_context: ctx,
        kv_f16,
        kv_budget_bytes: None,
        backend,
        cpu_fallback,
        experimental_moe,
    })
}

fn env_cpu_fallback() -> bool {
    std::env::var("ARTIFICER_CPU_FALLBACK").is_ok_and(|v| v == "1")
}

fn log_load_banner(engine: &Engine, load_s: f32) {
    let mem = SystemMemory::detect();
    eprintln!(
        "loaded in {:.2}s  (backend={}, arch={}, {} layers, embd={}, vocab={}, n_ctx={}, avx2={}, avx512_vnni={}, pool={})",
        load_s,
        engine.backend_name(),
        engine.model().config.arch_str,
        engine.model().config.n_layer,
        engine.model().config.n_embd,
        engine.model().config.n_vocab,
        engine.model().config.n_ctx,
        avx2_enabled(),
        avx512_vnni_enabled(),
        pool_workers(),
    );
    if let Some(w) = engine.moe_warning() {
        eprintln!("warning: {w}");
    }
    if let Some(r) = engine.context_report {
        if r.capped {
            eprintln!("{}", context_capped_line(&r));
        }
    }
    if let Some(line) = host_ram_line(&mem, engine.kv_storage_bytes() as u64) {
        eprintln!("{line}");
    }
}

/// `host RAM: …`, or `None` when total RAM is unknown. Available RAM of 0 means not detected
/// (macOS reads only the total): print "unknown", never a made-up 0.0 GiB.
fn host_ram_line(mem: &SystemMemory, kv_storage_bytes: u64) -> Option<String> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if mem.total_bytes == 0 {
        return None;
    }
    let available = if mem.available_bytes > 0 {
        format!("{:.1} GiB", mem.available_bytes as f64 / GIB)
    } else {
        "unknown".to_string()
    };
    Some(format!(
        "host RAM: total {:.1} GiB, available {available}; KV storage {:.1} MiB",
        mem.total_bytes as f64 / GIB,
        kv_storage_bytes as f64 / (1024.0 * 1024.0),
    ))
}

/// `context capped: …`. RAM not detected leaves the budget at `u64::MAX`: say so instead of
/// printing it as MiB.
fn context_capped_line(r: &artificer_core::ContextBudgetReport) -> String {
    let budget = if r.budget_bytes == u64::MAX {
        "budget unknown: RAM not detected, capped by --ctx".to_string()
    } else {
        format!(
            "budget {:.1} MiB",
            r.budget_bytes as f64 / (1024.0 * 1024.0)
        )
    };
    format!(
        "context capped: GGUF asked for {} → effective {} (KV ~{:.1} MiB, {budget})",
        r.requested_ctx,
        r.effective_ctx,
        r.kv_bytes as f64 / (1024.0 * 1024.0),
    )
}

fn inspect(path: PathBuf, show_meta: bool, show_tensors: bool, no_mmap: bool) -> Result<()> {
    let opts = LoadOptions {
        mode: if no_mmap {
            LoadMode::Buffered
        } else {
            LoadMode::Mmap
        },
        validate_tensor_bounds: true,
    };

    let model = GgufModel::load_with(&path, opts)
        .with_context(|| format!("failed to load GGUF model `{}`", path.display()))?;

    print!("{}", model.summary());

    if show_meta {
        println!("\nMetadata:");
        for (k, v) in model.metadata().iter() {
            println!("  {k} = {v}");
        }
    }

    if show_tensors {
        println!("\nTensors ({}):", model.tensor_count());
        for t in model.tensors() {
            let nbytes = t
                .nbytes()
                .map(|n| format!("{n}"))
                .unwrap_or_else(|| "?".into());
            let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
            println!(
                "  {:40}  {:>8}  [{:>}]  {} bytes  (data+{})",
                t.name,
                t.tensor_type,
                shape.join(", "),
                nbytes,
                t.offset
            );
        }
    }

    Ok(())
}

/// Context for every model-load failure (`run`, `bench`, `perplexity`), so refusals read the same.
fn load_failed(path: &std::path::Path) -> String {
    format!("failed to load model `{}`", path.display())
}

fn cmd_run(
    path: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: Option<u64>,
    stop: Vec<String>,
    raw: bool,
    opts: EngineLoadOptions,
) -> Result<()> {
    eprintln!("loading {} ...", path.display());
    let t0 = Instant::now();
    let mut engine = Engine::load_with(&path, opts).with_context(|| load_failed(&path))?;
    log_load_banner(&engine, t0.elapsed().as_secs_f32());

    // `--raw`: the caller supplies the whole template, so its control tokens are parsed.
    // Otherwise the prompt is user text and can never become a special token.
    let formatted = if raw {
        RenderedPrompt::raw(&prompt)
    } else {
        render_user_chat_prompt(&engine.model().tokenizer, &prompt)
            .context("chat template formatting failed")?
    };
    let template_src = formatted.source();

    let mut stop_strings = stop;
    if stop_strings.is_empty() && !raw {
        stop_strings.extend(["\nUSER:".into(), "\nUser:".into(), "\nASSISTANT:".into()]);
    }
    // Always include ChatML controls; runtime also merges these, but listing them
    // here keeps CLI `--stop` behavior obvious for raw Nemotron-style prompts.
    for s in artificer_runtime::CHAT_CONTROL_STOP_STRINGS {
        if !stop_strings.iter().any(|x| x == *s) {
            stop_strings.push((*s).to_owned());
        }
    }

    let cfg = GenerateConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop_on_eos: true,
        seed,
        stop_token_ids: Vec::new(),
        stop_strings,
    };

    eprintln!("user: {prompt:?}");
    match formatted.template_error() {
        // Loud: the model is not getting its own chat format.
        Some(e) => {
            eprintln!("template: {template_src:?} (WARNING: GGUF chat_template failed: {e})")
        }
        None => eprintln!("template: {template_src:?}"),
    }
    let t1 = Instant::now();
    let out = engine
        .generate_prompt(&formatted, &cfg)
        .context("generation failed")?;
    let elapsed = t1.elapsed().as_secs_f32();
    let n_new = out.completion_tokens.len();
    let tps = if elapsed > 0.0 {
        n_new as f32 / elapsed
    } else {
        0.0
    };

    eprintln!(
        "generated {n_new} tokens in {elapsed:.2}s ({tps:.2} tok/s); prompt tokens={}; finish={:?}",
        out.prompt_tokens.len(),
        out.finish_reason
    );
    println!("{}", out.text.trim());
    Ok(())
}

fn cmd_bench(
    path: PathBuf,
    prompt: String,
    max_tokens: usize,
    opts: EngineLoadOptions,
    runs: u32,
    warmup: u32,
) -> Result<()> {
    eprintln!("loading {} ...", path.display());
    let t0 = Instant::now();
    let mut engine = Engine::load_with(&path, opts).with_context(|| load_failed(&path))?;
    log_load_banner(&engine, t0.elapsed().as_secs_f32());

    let formatted = render_user_chat_prompt(&engine.model().tokenizer, &prompt)
        .context("chat template formatting failed")?;
    let cfg = GenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        stop_on_eos: true,
        seed: Some(1),
        stop_token_ids: Vec::new(),
        stop_strings: vec!["\nUSER:".into(), "\nUser:".into(), "\nASSISTANT:".into()],
    };

    // Warm-up generates (not timed), same process / same engine as the timed runs.
    for _ in 0..warmup {
        let _ = engine
            .generate_prompt(&formatted, &cfg)
            .context("warmup failed")?;
    }
    if profile_env_enabled() {
        profile_reset();
    }

    let mut total_prompt = 0usize;
    let mut total_gen = 0usize;
    let mut total_s = 0.0f32;
    let mut total_decode_s = 0.0f32;
    for i in 0..runs.max(1) {
        let t1 = Instant::now();
        let mut first = true;
        let mut t_first = Instant::now();
        let out = engine
            .generate_prompt_with_callback(&formatted, &cfg, |_| {
                if first {
                    t_first = Instant::now();
                    first = false;
                }
            })
            .with_context(|| format!("bench run {i} failed"))?;
        let elapsed = t1.elapsed().as_secs_f32();
        let decode_s = if out.completion_tokens.is_empty() {
            0.0
        } else {
            t_first.elapsed().as_secs_f32()
        };
        total_prompt += out.prompt_tokens.len();
        total_gen += out.completion_tokens.len();
        total_s += elapsed;
        total_decode_s += decode_s;
        let e2e = if elapsed > 0.0 {
            out.completion_tokens.len() as f32 / elapsed
        } else {
            0.0
        };
        let dec = if decode_s > 0.0 {
            out.completion_tokens.len() as f32 / decode_s
        } else {
            0.0
        };
        let t = out.timings;
        eprintln!(
            "run {}: prompt={} gen={} wall={:.3}s (e2e {:.2} gen-tok/s) decode≈{:.3}s ({:.2} tok/s) engine_prefill={:.3}ms ({:.2} tok/s) engine_decode={:.3}ms ({:.2} tok/s)",
            i + 1,
            out.prompt_tokens.len(),
            out.completion_tokens.len(),
            elapsed,
            e2e,
            decode_s,
            dec,
            t.prefill_ms,
            t.prefill_tok_s().unwrap_or(0.0),
            t.decode_ms,
            t.decode_tok_s().unwrap_or(0.0)
        );
    }
    let avg_e2e = if total_s > 0.0 {
        total_gen as f32 / total_s
    } else {
        0.0
    };
    let avg_dec = if total_decode_s > 0.0 {
        total_gen as f32 / total_decode_s
    } else {
        0.0
    };
    eprintln!("--- bench summary ---");
    eprintln!("avg e2e gen-tok/s (includes prefill): {avg_e2e:.2}");
    eprintln!("avg decode tok/s (after first token): {avg_dec:.2}");
    eprintln!(
        "total prompt tokens: {total_prompt}, total gen tokens: {total_gen}, wall: {total_s:.3}s"
    );
    if let Some(rss) = peak_rss_bytes() {
        eprintln!(
            "peak RSS (VmHWM): {:.1} MiB",
            rss as f64 / (1024.0 * 1024.0)
        );
    }
    if let Some(rss) = current_rss_bytes() {
        eprintln!("current RSS: {:.1} MiB", rss as f64 / (1024.0 * 1024.0));
    }
    eprintln!(
        "avx2: {}  avx512: {}  avx512_vnni: {}  pool: {}",
        avx2_enabled(),
        avx512_enabled(),
        avx512_vnni_enabled(),
        pool_workers()
    );
    if profile_env_enabled() {
        profile_dump("bench-decode");
    }
    Ok(())
}

/// CLI security settings for `serve`. Environment is read only here, not by library callers
/// (so embedded servers and tests keep their explicit defaults).
///
/// `api_key` is a raw `String` so clap does not parse it through [`artificer_server::ApiKey`]'s
/// `FromStr` (that error path echoes the value). Validation happens in [`ServeSecurityArgs::apply`].
#[derive(clap::Args, Default)]
struct ServeSecurityArgs {
    /// Extra Host names (comma-separated), added to loopback defaults. `*` disables the
    /// Host check. Also ARTIFICER_ALLOWED_HOSTS.
    #[arg(long, value_name = "HOSTS")]
    allowed_hosts: Option<String>,
    /// Extra Origin values (comma-separated), added to loopback defaults. `*` disables
    /// the Origin check. Also ARTIFICER_ALLOWED_ORIGINS.
    #[arg(long, value_name = "ORIGINS")]
    allowed_origins: Option<String>,
    /// Require Authorization: Bearer on every route except GET /health. Also ARTIFICER_API_KEY.
    #[arg(long, value_name = "KEY")]
    api_key: Option<String>,
    /// Allow a non-loopback --host without an API key (unsafe on an untrusted network).
    #[arg(long)]
    allow_insecure_bind: bool,
}

impl std::fmt::Debug for ServeSecurityArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeSecurityArgs")
            .field("allowed_hosts", &self.allowed_hosts)
            .field("allowed_origins", &self.allowed_origins)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("allow_insecure_bind", &self.allow_insecure_bind)
            .finish()
    }
}

impl ServeSecurityArgs {
    fn apply(self, cfg: &mut ServerConfig) -> Result<()> {
        let env = |name: &str| -> Result<Option<String>> {
            match std::env::var(name) {
                Ok(v) => Ok(Some(v)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err(anyhow::anyhow!("{name} must be valid UTF-8"))
                }
            }
        };
        let list = |v: Option<String>| -> Vec<String> {
            v.unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        };
        cfg.allowed_hosts = list(match self.allowed_hosts {
            Some(v) => Some(v),
            None => env("ARTIFICER_ALLOWED_HOSTS")?,
        });
        cfg.allowed_origins = list(match self.allowed_origins {
            Some(v) => Some(v),
            None => env("ARTIFICER_ALLOWED_ORIGINS")?,
        });
        cfg.api_key =
            match self.api_key {
                Some(v) => Some(v.parse::<artificer_server::ApiKey>().map_err(|e| {
                    anyhow::anyhow!("{}", e.replacen("the API key", "--api-key", 1))
                })?),
                None => match env("ARTIFICER_API_KEY")? {
                    None => None,
                    Some(v) => Some(v.parse::<artificer_server::ApiKey>().map_err(|e| {
                        anyhow::anyhow!("{}", e.replacen("the API key", "ARTIFICER_API_KEY", 1))
                    })?),
                },
            };
        cfg.allow_insecure_bind = self.allow_insecure_bind;
        Ok(())
    }
}

// Keep the no-extra-options entry point for the existing bind-first CLI tests.
#[cfg(test)]
fn cmd_serve(
    path: PathBuf,
    host: String,
    port: u16,
    model_id: Option<String>,
    opts: EngineLoadOptions,
) -> Result<()> {
    cmd_serve_with_security(
        path,
        host,
        port,
        model_id,
        opts,
        ServeSecurityArgs::default(),
    )
}

fn cmd_serve_with_security(
    path: PathBuf,
    host: String,
    port: u16,
    model_id: Option<String>,
    opts: EngineLoadOptions,
    security: ServeSecurityArgs,
) -> Result<()> {
    let model_id = model_id.unwrap_or_else(|| default_model_id(&path));
    let mut cfg = ServerConfig {
        host,
        port,
        model_path: path,
        model_id,
        ..ServerConfig::default()
    };
    security.apply(&mut cfg)?;
    // Stash load opts via environment-less path: ServerConfig extension.
    cfg.load_opts = Some(opts);
    match serve_blocking(cfg) {
        // Already one plain line (address, OS error, what to try): print it without a chain.
        Err(e @ ServerError::Bind { .. }) => Err(anyhow::anyhow!("{e}")),
        other => other.context("server failed"),
    }
}

fn cmd_mcp(path: PathBuf, model_id: Option<String>, opts: EngineLoadOptions) -> Result<()> {
    let model_id = model_id.unwrap_or_else(|| default_model_id(&path));
    serve_mcp_stdio_blocking(&path, model_id, opts).context("MCP server failed")
}

#[allow(clippy::too_many_arguments)]
fn cmd_perplexity(
    path: PathBuf,
    file: PathBuf,
    ctx: u32,
    chunks: Option<usize>,
    threads: Option<usize>,
    kv_f16: bool,
    backend_flag: Option<&str>,
    dump_tokens: Option<PathBuf>,
    experimental_moe: bool,
) -> Result<()> {
    use artificer_core::perplexity::{run_perplexity, PplPlan};
    use artificer_core::{create_backend, KvElem, LlamaModel};

    // The matmul pool is created lazily on first use, so this must run before any work.
    if let Some(t) = threads {
        anyhow::ensure!((1..=64).contains(&t), "--threads must be 1..=64");
        std::env::set_var("ARTIFICER_POOL", t.to_string());
    }
    let kind = resolve_backend_kind(backend_flag)?;
    if kind == BackendKind::Cuda && !cuda_feature_enabled() {
        anyhow::bail!("CUDA backend requested but this binary was built without `--features cuda`");
    }

    let bytes = std::fs::read(&file).with_context(|| format!("read {}", file.display()))?;
    let mut text =
        String::from_utf8(bytes).with_context(|| format!("{} is not UTF-8", file.display()))?;
    // llama.cpp `-f` (common/arg.cpp): drop one trailing newline.
    if text.ends_with('\n') {
        text.pop();
    }

    eprintln!("loading {} ...", path.display());
    let t0 = Instant::now();
    let mut model = LlamaModel::load(&path).with_context(|| load_failed(&path))?;
    model.config.n_ctx = ctx;
    // Same order as `Engine`: the CUDA refusal (MoE included) wins over the MoE flag.
    // Both refusals carry the same "failed to load model" context as `run`.
    if kind == BackendKind::Cuda {
        model
            .check_cuda_supported()
            .with_context(|| load_failed(&path))?;
    }
    if let Some(w) = model
        .config
        .check_moe_gate(experimental_moe)
        .with_context(|| load_failed(&path))?
    {
        eprintln!("warning: {w}");
    }
    let be = create_backend(kind)?;
    eprintln!(
        "loaded in {:.2}s (backend={kind:?}, arch={}, vocab={}, pool={})",
        t0.elapsed().as_secs_f32(),
        model.config.arch_str,
        model.config.n_vocab,
        pool_workers()
    );

    // llama.cpp: common_tokenize(ctx, prompt, add_special = true, parse_special = false).
    let t1 = Instant::now();
    let tokens = model.tokenizer.encode_with(&text, true, false)?;
    eprintln!(
        "perplexity: tokenized {} bytes -> {} tokens in {:.1} ms (add_bos={}, bos_id={})",
        text.len(),
        tokens.len(),
        t1.elapsed().as_secs_f64() * 1e3,
        model.tokenizer.add_bos,
        model.tokenizer.bos_id
    );
    if let Some(p) = &dump_tokens {
        let s: String = tokens.iter().map(|t| format!("{t}\n")).collect();
        std::fs::write(p, s).with_context(|| format!("write {}", p.display()))?;
    }
    let plan = PplPlan::new(tokens.len(), ctx as usize, chunks)?;
    eprintln!(
        "perplexity: calculating perplexity over {} chunks, n_ctx={}, scored tokens {} ({} per chunk, positions {}..{})",
        plan.n_chunk,
        plan.n_ctx,
        plan.scored_total(),
        plan.scored_per_chunk(),
        plan.scored_positions().start,
        plan.scored_positions().end
    );
    let kv = if kv_f16 { KvElem::F16 } else { KvElem::F32 };
    let t2 = Instant::now();
    let mut out = std::io::stdout();
    let r = run_perplexity(&model, be.as_ref(), &tokens, &plan, kv, &mut out)?;
    let secs = t2.elapsed().as_secs_f64();
    match r.stderr {
        Some(se) => println!("Final estimate: PPL = {:.4} +/- {:.5}", r.ppl, se),
        None => println!(
            "Final estimate: PPL = {:.4} (non-positive variance; no stderr)",
            r.ppl
        ),
    }
    eprintln!(
        "perplexity: scored {} tokens in {secs:.2}s ({:.2} s/chunk)",
        r.accum.count,
        secs / plan.n_chunk.max(1) as f64
    );
    Ok(())
}

fn default_model_id(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local-model".into())
}

#[cfg(test)]
mod tests {
    use super::{
        cmd_serve, context_capped_line, host_ram_line, load_failed, load_opts, Cli, Commands,
    };
    use anyhow::Context;
    use artificer_core::SystemMemory;
    use clap::Parser;

    fn report(budget_bytes: u64) -> artificer_core::ContextBudgetReport {
        artificer_core::ContextBudgetReport {
            requested_ctx: 32768,
            effective_ctx: 8192,
            budget_bytes,
            kv_bytes: 512 << 20,
            total_ram_bytes: 0,
            available_ram_bytes: 0,
            capped: true,
        }
    }

    #[test]
    fn host_ram_line_says_unknown_for_undetected_available_ram() {
        let mem = |total_bytes, available_bytes| SystemMemory {
            total_bytes,
            available_bytes,
        };
        // macOS: total only.
        assert_eq!(
            host_ram_line(&mem(16 << 30, 0), 64 << 20).as_deref(),
            Some("host RAM: total 16.0 GiB, available unknown; KV storage 64.0 MiB")
        );
        // Linux / Windows: both read.
        assert_eq!(
            host_ram_line(&mem(16 << 30, 6 << 30), 64 << 20).as_deref(),
            Some("host RAM: total 16.0 GiB, available 6.0 GiB; KV storage 64.0 MiB")
        );
        assert_eq!(host_ram_line(&mem(0, 0), 64 << 20), None);
    }

    #[test]
    fn unknown_ram_never_prints_the_u64_max_budget() {
        let line = context_capped_line(&report(u64::MAX));
        assert_eq!(
            line,
            "context capped: GGUF asked for 32768 → effective 8192 (KV ~512.0 MiB, budget unknown: RAM not detected, capped by --ctx)"
        );
        assert!(!line.contains("17592186044416"), "{line}");
        let known = context_capped_line(&report(3 << 30));
        assert!(known.ends_with("budget 3072.0 MiB)"), "{known}");
    }

    #[test]
    fn busy_port_is_one_plain_line_before_the_model_is_read() {
        let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = held.local_addr().unwrap().port();
        let e = cmd_serve(
            std::path::PathBuf::from("/nonexistent/artificer-cli-bind.gguf"),
            "127.0.0.1".into(),
            port,
            None,
            artificer_runtime::EngineLoadOptions::default(),
        )
        .unwrap_err();
        // What main prints: `error: {e:#}`.
        let line = format!("{e:#}");
        assert!(
            line.starts_with(&format!("could not listen on 127.0.0.1:{port} (")),
            "{line}"
        );
        assert!(line.ends_with("try --port <N>."), "{line}");
        assert!(line.contains("already in use"), "{line}");
        assert!(!line.contains("server failed"), "{line}");
        assert!(!line.contains("artificer-cli-bind"), "{line}");
        assert!(!line.contains('\n'), "{line}");
        drop(held);
    }

    #[test]
    fn free_port_and_missing_model_is_a_load_error_and_frees_the_port() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let e = cmd_serve(
            std::path::PathBuf::from("/nonexistent/artificer-cli-load.gguf"),
            "127.0.0.1".into(),
            port,
            None,
            artificer_runtime::EngineLoadOptions::default(),
        )
        .unwrap_err();
        let text = format!("{e:#}");
        assert!(text.starts_with("server failed: "), "{text}");
        assert!(!text.contains("could not listen"), "{text}");
        // The bound listener is dropped with the failed load: the port is free again.
        std::net::TcpListener::bind(("127.0.0.1", port))
            .unwrap_or_else(|err| panic!("port {port} still held after the load error: {err}"));
    }

    #[test]
    fn load_refusal_carries_failed_to_load_context() {
        let path = std::path::PathBuf::from("m.gguf");
        let gate = "`olmoe` is an MoE model; MoE support is experimental: pass --experimental-moe to load it";
        let r: Result<(), _> = Err(artificer_core::CoreError::Other(gate.into()));
        let e = r.with_context(|| load_failed(&path)).unwrap_err();
        assert_eq!(
            format!("{e:#}"),
            format!("failed to load model `m.gguf`: {gate}")
        );
    }

    fn experimental_moe_of(args: &[&str]) -> bool {
        match Cli::try_parse_from(args.iter().copied()).unwrap().command {
            Commands::Run {
                experimental_moe, ..
            }
            | Commands::Bench {
                experimental_moe, ..
            }
            | Commands::Serve {
                experimental_moe, ..
            }
            | Commands::Mcp {
                experimental_moe, ..
            }
            | Commands::Perplexity {
                experimental_moe, ..
            } => experimental_moe,
            Commands::Inspect { .. } => panic!("inspect has no --experimental-moe"),
        }
    }

    #[test]
    fn experimental_moe_flag_parses_on_every_loading_command() {
        let cmds: [&[&str]; 5] = [
            &["artificer", "run", "m.gguf", "-p", "hi"],
            &["artificer", "bench", "m.gguf"],
            &["artificer", "serve", "m.gguf"],
            &["artificer", "mcp", "m.gguf"],
            &["artificer", "perplexity", "m.gguf", "-f", "t.txt"],
        ];
        for base in cmds {
            assert!(!experimental_moe_of(base), "{base:?} default");
            let mut with: Vec<&str> = base.to_vec();
            with.push("--experimental-moe");
            assert!(experimental_moe_of(&with), "{with:?}");
        }
    }

    #[test]
    fn load_opts_carries_experimental_moe() {
        for flag in [false, true] {
            let o = load_opts(8192, false, false, Some("cpu"), false, flag).unwrap();
            assert_eq!(o.experimental_moe, flag);
        }
    }
}
