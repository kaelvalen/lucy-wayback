{
  description = "Lucy: sparse, event-driven, CPU-only learning substrate";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs { inherit system; overlays = [ rust-overlay.overlays.default ]; }));
    in {
      devShells = forAll (pkgs: {
        # Toolchain comes from rust-toolchain.toml so rustup users and nix users get the same compiler.
        default = pkgs.mkShell {
          packages = [
            (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            (pkgs.python3.withPackages (ps: [ ps.matplotlib ]))
          ];
        };
      });
    };
}
