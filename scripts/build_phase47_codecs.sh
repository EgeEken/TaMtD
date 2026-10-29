#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${TAMTD_PHASE47_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tamtd/phase47-codecs}"
generator="${TAMTD_PHASE47_CMAKE_GENERATOR:-Ninja}"
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
mkdir -p "$build_root"

clone_pinned() {
  local name="$1" repository="$2" commit="$3"
  local source="$build_root/$name"
  if [[ ! -d "$source/.git" ]]; then
    git clone --no-checkout "$repository" "$source"
  fi
  git -C "$source" fetch origin "$commit"
  git -C "$source" checkout --detach "$commit"
  git -C "$source" submodule update --init --recursive
  local actual
  actual="$(git -C "$source" rev-parse HEAD)"
  [[ "$actual" == "$commit" ]] || { echo "$name revision mismatch: $actual" >&2; exit 1; }
}

apply_patch_once() {
  local name="$1" patch="$repo_root/patches/phase47/$2"
  local source="$build_root/$name"
  if git -C "$source" apply --unidiff-zero --reverse --check "$patch" >/dev/null 2>&1; then
    return
  fi
  git -C "$source" apply --unidiff-zero --check "$patch"
  git -C "$source" apply --unidiff-zero "$patch"
}

cmake_build() {
  local source="$1" build="$2"
  shift 2
  cmake -S "$source" -B "$build" -G "$generator" -DCMAKE_BUILD_TYPE=Release "$@"
  cmake --build "$build" --parallel "$jobs"
}

clone_pinned libjxl https://github.com/libjxl/libjxl.git a7a9c787341cf703dede03c2009fa460cae5e5df
clone_pinned libwebp https://github.com/webmproject/libwebp.git 4fa21912338357f89e4fd51cf2368325b59e9bd9
clone_pinned libaom https://aomedia.googlesource.com/aom 44d0a57786f432d933ff64b653347c66f4d0fa1d
clone_pinned libavif https://github.com/AOMediaCodec/libavif.git c5240fc79fe5c2407e10afd35f5505ef6333ea49
apply_patch_once libwebp libwebp-pre-reconstruction.patch
apply_patch_once libaom libaom-pre-reconstruction.patch

cmake_build "$build_root/libjxl" "$build_root/build-libjxl" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
  -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
  -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_PLUGINS=OFF

cmake_build "$build_root/libwebp" "$build_root/build-libwebp" \
  -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF \
  -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF \
  -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_TESTS=OFF \
  -DWEBP_BUILD_LIBWEBPMUX=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF \
  -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_WEBP_JS=OFF

cmake_build "$build_root/libaom" "$build_root/build-libaom" \
  -DBUILD_SHARED_LIBS=ON -DCONFIG_AV1_DECODER=1 -DCONFIG_AV1_ENCODER=0 \
  -DCONFIG_MULTITHREAD=0 -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=0

if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
  aom_library="$build_root/build-libaom/libaom.dll.a"
else
  aom_library="$build_root/build-libaom/libaom.so"
fi
cmake_build "$build_root/libavif" "$build_root/build-libavif" \
  -DBUILD_SHARED_LIBS=ON -DAVIF_CODEC_AOM=SYSTEM \
  -DAVIF_CODEC_AOM_DECODE=ON -DAVIF_CODEC_AOM_ENCODE=OFF \
  -DAVIF_LIBYUV=OFF -DAVIF_BUILD_APPS=OFF -DAVIF_BUILD_TESTS=OFF \
  -DAOM_INCLUDE_DIR="$build_root/libaom" -DAOM_LIBRARY="$aom_library"

printf 'Phase 4.7 decoder builds are ready at: %s\n' "$build_root"
