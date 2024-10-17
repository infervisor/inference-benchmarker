use std::error::Error;
use vergen_gitcl::{Emitter, GitclBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    // Try to get the git sha from the local git repository
    let gitcl = match GitclBuilder::all_git() {
        Ok(gitcl) => gitcl,
        Err(_) => {
            fallback_git_sha();
            return Ok(());
        }
    };
    if Emitter::default()
        .fail_on_error()
        .add_instructions(&gitcl)?
        .emit()
        .is_err()
    {
        fallback_git_sha();
    }
    Ok(())
}

fn fallback_git_sha() {
    // Unable to get the git sha
    if let Ok(sha) = std::env::var("GIT_SHA") {
        // Set it from an env var
        println!("cargo:rustc-env=VERGEN_GIT_SHA={sha}");
    }
}
