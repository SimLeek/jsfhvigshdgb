use std::env;
use std::fs;

fn is_program_in_path(program: &str) -> bool {
    // thanks: https://stackoverflow.com/a/35046243/782170
    if let Ok(path) = env::var("PATH") {
        for p in path.split(":") {
            let p_str = format!("{}/{}", p, program);
            if fs::metadata(p_str).is_ok() {
                return true;
            }
        }
    }
    false
}

#[cfg(windows)]
fn main() {
    let program = "ninja";
    if is_program_in_path(program) {
        return;
    } else {
        println!("Ninja is not installed. If building with msvc, please install: https://github.com/ninja-build/ninja/releases");
        return;
    }
}

#[cfg(not(windows))]
fn main() {
    return;
}