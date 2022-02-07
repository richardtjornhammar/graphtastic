with (import <nixpkgs> {}).pkgs;
with lib;

let
  myPyPkgs = python310Packages.override {
    overrides = self: super: {
      graphtastic = super.buildPythonPackage rec {
        pname = "graphtastic";
        version = "0.10.5";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "0jrcw7vnmp4k9svksyp052lqvgg7fd9w3kar9agsl3hmni87gnal";
        };
        buildInputs = with super;
          [ numpy scipy ];
      };

    };
  };
in

stdenv.mkDerivation rec {
  name = "graphtastic";
  buildInputs = (with myPyPkgs;
    [
      python numpy graphtastic
    ]);
  src = null;
  shellHook = ''
    # Allow the use of wheels.
    SOURCE_DATE_EPOCH=$(date +%s)

    # Augment the dynamic linker path
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${R}/lib/R/lib:${readline}/lib

    echo "********************************"
    echo "* WELCOME TO ${toUpper name} SHELL *"
    echo "********************************"
  '';

}
