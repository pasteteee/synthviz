#!/usr/bin/env bash
# Optional Surge XT layer (GPL-3). Two ways:
#   1) prebuilt:  SURGE_PREBUILT=/path/with/lib_and_data bash third_party/surge/install.sh
#   2) from source (~20-30 min): bash third_party/surge/install.sh
# Result: third_party/surge/lib/surgepy*.so and factory data COPIED to /usr/local/share/surge-xt
# (must be a local disk: on a network mount Surge needs ~90 s to start).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
if [ -n "$SURGE_PREBUILT" ]; then
  mkdir -p "$HERE/lib"; cp -r "$SURGE_PREBUILT/lib/." "$HERE/lib/"
  DATA="$SURGE_PREBUILT/data"
else
  $SUDO apt-get install -y -qq git cmake build-essential libasound2-dev libjack-jackd2-dev libcairo2-dev \
      libxkbcommon-x11-dev libxcb-cursor-dev libfreetype6-dev libfontconfig1-dev python3-dev >/dev/null
  SRC="${TMPDIR:-/tmp}/surge-src"
  [ -d "$SRC" ] || git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/surge-synthesizer/surge "$SRC"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DSURGE_BUILD_PYTHON_BINDINGS=ON -DSURGE_SKIP_JUCE_FOR_RACK=ON
  cmake --build "$SRC/build" --target surgepy --parallel "$(nproc)"
  mkdir -p "$HERE/lib"; cp "$SRC"/build/src/surge-python/surgepy*.so "$HERE/lib/"
  DATA="$SRC/resources/data"
fi
$SUDO rm -rf /usr/local/share/surge-xt
$SUDO cp -r "$DATA" /usr/local/share/surge-xt
# surge_render.py looks for data next to itself
ln -sfn /usr/local/share/surge-xt "$HERE/data"
python3 -c "import sys; sys.path.insert(0, '$HERE'); import surge_render as r; print('surge ok:', len(r.list_patches()), 'patches')"
