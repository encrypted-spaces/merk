fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(use_box)");
    let is_box = std::env::var("CARGO_FEATURE_BOX").is_ok();
    let is_zkvm = std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("zkvm");
    if is_box || is_zkvm {
        println!("cargo:rustc-cfg=use_box");
    }
}
