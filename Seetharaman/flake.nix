{
  inputs = {
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db/4be31ec67b9283eac29494920588a9748eb06d18";
      flake = false;
    };
    mach-nix = {
      url = "mach-nix/3.5.0";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
    };
  };

  outputs = {self, nixpkgs, mach-nix, pypi-deps-db }@inp:
    let

      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems (system: f system (import nixpkgs {inherit system;}));

    in
    {
      devShell = forAllSystems (system: pkgs: mach-nix.lib."${system}".mkPythonShell {
        requirements = ''
          numpy
          scipy
          matplotlib
          sympy
          gmpy2
          tqdm
	  numba
	  numdifftools
	  lmfit
        '';
      });
    };
}

