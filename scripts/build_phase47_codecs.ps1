param(
    [string]$BuildRoot = $env:TAMTD_PHASE47_BUILD_DIR,
    [string]$Generator = $env:TAMTD_PHASE47_CMAKE_GENERATOR,
    [int]$Jobs = 4
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildRoot) { $BuildRoot = Join-Path $env:TEMP 'tamtd-phase47-codecs' }
if (-not $Generator) { $Generator = 'Ninja' }
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}

function Clone-Pinned([string]$Name, [string]$Repository, [string]$Commit) {
    $source = Join-Path $BuildRoot $Name
    if (-not (Test-Path (Join-Path $source '.git'))) {
        Invoke-Checked 'git' @('clone', '--no-checkout', $Repository, $source)
    }
    Invoke-Checked 'git' @('-C', $source, 'fetch', 'origin', $Commit)
    Invoke-Checked 'git' @('-C', $source, 'checkout', '--detach', $Commit)
    Invoke-Checked 'git' @('-C', $source, 'submodule', 'update', '--init', '--recursive')
    $actual = (& git -C $source rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Commit) { throw "$Name revision mismatch: $actual" }
}

function Apply-PatchOnce([string]$Name, [string]$PatchName) {
    $source = Join-Path $BuildRoot $Name
    $patch = Join-Path $repoRoot (Join-Path 'patches/phase47' $PatchName)
    & git -C $source apply --unidiff-zero --reverse --check $patch 2>$null
    if ($LASTEXITCODE -eq 0) { return }
    Invoke-Checked 'git' @('-C', $source, 'apply', '--unidiff-zero', '--check', $patch)
    Invoke-Checked 'git' @('-C', $source, 'apply', '--unidiff-zero', $patch)
}

function Build-CMake([string]$Name, [string[]]$Options) {
    $source = Join-Path $BuildRoot $Name
    $build = Join-Path $BuildRoot "build-$Name"
    Invoke-Checked 'cmake' (@('-S', $source, '-B', $build, '-G', $Generator, '-DCMAKE_BUILD_TYPE=Release') + $Options)
    Invoke-Checked 'cmake' @('--build', $build, '--parallel', "$Jobs")
}

Clone-Pinned 'libjxl' 'https://github.com/libjxl/libjxl.git' 'a7a9c787341cf703dede03c2009fa460cae5e5df'
Clone-Pinned 'libwebp' 'https://github.com/webmproject/libwebp.git' '4fa21912338357f89e4fd51cf2368325b59e9bd9'
Clone-Pinned 'libaom' 'https://aomedia.googlesource.com/aom' '44d0a57786f432d933ff64b653347c66f4d0fa1d'
Clone-Pinned 'libavif' 'https://github.com/AOMediaCodec/libavif.git' 'c5240fc79fe5c2407e10afd35f5505ef6333ea49'
Apply-PatchOnce 'libwebp' 'libwebp-pre-reconstruction.patch'
Apply-PatchOnce 'libaom' 'libaom-pre-reconstruction.patch'

Build-CMake 'libjxl' @('-DBUILD_SHARED_LIBS=ON', '-DBUILD_TESTING=OFF', '-DJPEGXL_ENABLE_TOOLS=OFF', '-DJPEGXL_ENABLE_EXAMPLES=OFF', '-DJPEGXL_ENABLE_BENCHMARK=OFF', '-DJPEGXL_ENABLE_MANPAGES=OFF', '-DJPEGXL_ENABLE_OPENEXR=OFF', '-DJPEGXL_ENABLE_JNI=OFF', '-DJPEGXL_ENABLE_PLUGINS=OFF')
Build-CMake 'libwebp' @('-DBUILD_SHARED_LIBS=ON', '-DWEBP_BUILD_ANIM_UTILS=OFF', '-DWEBP_BUILD_CWEBP=OFF', '-DWEBP_BUILD_DWEBP=OFF', '-DWEBP_BUILD_EXTRAS=OFF', '-DWEBP_BUILD_GIF2WEBP=OFF', '-DWEBP_BUILD_IMG2WEBP=OFF', '-DWEBP_BUILD_LIBWEBPMUX=OFF', '-DWEBP_BUILD_TESTS=OFF', '-DWEBP_BUILD_VWEBP=OFF', '-DWEBP_BUILD_WEBPINFO=OFF', '-DWEBP_BUILD_WEBPMUX=OFF', '-DWEBP_BUILD_WEBP_JS=OFF')
Build-CMake 'libaom' @('-DBUILD_SHARED_LIBS=ON', '-DCONFIG_AV1_DECODER=1', '-DCONFIG_AV1_ENCODER=0', '-DCONFIG_MULTITHREAD=0', '-DENABLE_TESTS=0', '-DENABLE_DOCS=0', '-DENABLE_EXAMPLES=0')

$aomBuild = Join-Path $BuildRoot 'build-libaom'
$aomLibrary = if (Test-Path (Join-Path $aomBuild 'libaom.dll.a')) { Join-Path $aomBuild 'libaom.dll.a' } else { Join-Path $aomBuild 'aom.lib' }
Build-CMake 'libavif' @('-DBUILD_SHARED_LIBS=ON', '-DAVIF_CODEC_AOM=SYSTEM', '-DAVIF_CODEC_AOM_DECODE=ON', '-DAVIF_CODEC_AOM_ENCODE=OFF', '-DAVIF_LIBYUV=OFF', '-DAVIF_BUILD_APPS=OFF', '-DAVIF_BUILD_TESTS=OFF', "-DAOM_INCLUDE_DIR=$(Join-Path $BuildRoot 'libaom')", "-DAOM_LIBRARY=$aomLibrary")

Write-Output "Phase 4.7 decoder builds are ready at: $BuildRoot"
