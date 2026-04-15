use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    bitloops_platform_embeddings::run(bitloops_platform_embeddings::Cli::parse())
}
