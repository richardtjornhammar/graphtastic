with (import <nixpkgs> {}).pkgs;
with lib;

let
  myPyPkgs = python39Packages.override {
    overrides = self: super: {
      graphtastic = super.buildPythonPackage rec {
        pname = "graphtastic";
        version = "0.10.10";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "1fv251801c398swivrfv55i01y79g2vdwq7vxvzn70kc7csb8mq7";
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
      python numpy scipy graphtastic numba
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
