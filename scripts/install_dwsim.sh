#!/usr/bin/env bash
# Install DWSIM 9.0.5 for the refinery's DWSIM comparisons (Linux, amd64).
#
#   scripts/install_dwsim.sh [TARGET_DIR]
#
# Unpacks the DWSIM .deb into TARGET_DIR (default: ./.dwsim) WITHOUT
# installing the package (dpkg-deb -x, no root needed for this step), installs
# the .NET 8 runtime from apt (needs root) and pythonnet from pip, then prints
# the DWSIM_PATH to export. The generators in tests/refinery/reference/
# (dwsim_*_generate.py) and difflow.dwsim_import read DWSIM_PATH.
#
# Where it comes from: SourceForge. DWSIM's GitHub releases are the usual
# source, but github.com release downloads are blocked by the build box's
# egress proxy, and SourceForge carries the same .deb. Verified build:
#   dwsim_9.0.5-amd64.deb  221,039,948 bytes
#   sha256 52c041b1d659ea26e22750e8b7045c7bc68d3d95f6384abe28d07d967eafa20c
# (the .deb's own dependency list -- dotnet-runtime-8.0 >= 8.0.11,
# coinor-libipopt1v5 -- is not installed by dpkg-deb -x; only the .NET
# runtime is needed for the thermodynamics and flowsheet automation used
# here. Ipopt is only for DWSIM's Gibbs-minimization flash.)
#
# Tested with: Ubuntu 24.04, dotnet-runtime-8.0 8.0.31, pythonnet 3.2.0,
# Python 3.11.
set -euo pipefail

TARGET="${1:-$PWD/.dwsim}"
VERSION="9.0.5"
DEB="dwsim_${VERSION}-amd64.deb"
URL="https://sourceforge.net/projects/dwsim/files/DWSIM/DWSIM%209.0/${VERSION}/${DEB}/download"
SHA256="52c041b1d659ea26e22750e8b7045c7bc68d3d95f6384abe28d07d967eafa20c"

mkdir -p "$TARGET"
cd "$TARGET"

if [ ! -f "$DEB" ]; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$DEB" "$URL"
fi
echo "$SHA256  $DEB" | sha256sum -c -

echo "extracting into $TARGET/root"
dpkg-deb -x "$DEB" root
LIB="$TARGET/root/usr/local/lib/dwsim"
test -f "$LIB/DWSIM.Automation.dll" || { echo "no DWSIM.Automation.dll in $LIB" >&2; exit 1; }

if ! dotnet --list-runtimes 2>/dev/null | grep -q "Microsoft.NETCore.App 8\."; then
    echo "installing the .NET 8 runtime (apt)"
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get update
    $SUDO apt-get install -y dotnet-runtime-8.0
fi

python -m pip install "pythonnet>=3.0"

# Smoke: load CoreCLR (not Mono, pythonnet's Linux default) and DWSIM.
DWSIM_PATH="$LIB" python - <<'PY'
import os
from pythonnet import load
load("coreclr")
import clr
lib = os.environ["DWSIM_PATH"]
clr.AddReference(os.path.join(lib, "DWSIM.Automation.dll"))
from DWSIM.Automation import Automation3
print(Automation3().GetVersion())
PY

echo
echo "DWSIM ${VERSION} is ready. Use it with:"
echo "  export DWSIM_PATH=$LIB"
