# CI only (Windows): append --remap-path-prefix flags for the release builds to RUSTFLAGS via $GITHUB_ENV, so
# runner paths do not end up in the binaries. Same targets as scripts/ci-remap.sh, plus the forward-slash form of
# every Windows path:
#   $CARGO_HOME\registry\src  -> /cargo/registry/src
#   $RUSTUP_HOME              -> /rustup
#   workspace                 -> /artificer
# REMAP_WORKSPACE is the workflow's `${{ github.workspace }}` (passed as an env var by the step); when empty,
# $GITHUB_WORKSPACE is used. If $GITHUB_WORKSPACE differs from it, that path is remapped too.
$ErrorActionPreference = "Stop"
function Trim-Path([string]$p) {
  if ([string]::IsNullOrEmpty($p)) { return $p }
  return $p.TrimEnd('\', '/')
}
function Add-Remap([ref]$flags, [string]$from, [string]$to) {
  if ([string]::IsNullOrEmpty($from)) { return }
  $from = Trim-Path $from
  $flags.Value = "$($flags.Value) --remap-path-prefix=$from=$to"
  $fwd = $from -replace '\\', '/'
  if ($fwd -ne $from) {
    $flags.Value = "$($flags.Value) --remap-path-prefix=$fwd=$to"
  }
}
$cargoHome = if (-not [string]::IsNullOrEmpty($env:CARGO_HOME)) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
$rustupHome = if (-not [string]::IsNullOrEmpty($env:RUSTUP_HOME)) { $env:RUSTUP_HOME } else { Join-Path $env:USERPROFILE ".rustup" }
$cargoHome = Trim-Path $cargoHome
$cargoReg = if ($cargoHome -match '/') { "$cargoHome/registry/src" } else { "$cargoHome\registry\src" }
$workspace = $env:REMAP_WORKSPACE
if ([string]::IsNullOrEmpty($workspace)) { $workspace = $env:GITHUB_WORKSPACE }
$flags = $env:RUSTFLAGS
if ($null -eq $flags) { $flags = "" }
Add-Remap ([ref]$flags) $cargoReg "/cargo/registry/src"
Add-Remap ([ref]$flags) $rustupHome "/rustup"
Add-Remap ([ref]$flags) $workspace "/artificer"
$wsEnv = $env:GITHUB_WORKSPACE
if (-not [string]::IsNullOrEmpty($wsEnv) -and (Trim-Path $wsEnv) -ne (Trim-Path $workspace)) {
  Add-Remap ([ref]$flags) $wsEnv "/artificer"
}
[System.IO.File]::AppendAllText($env:GITHUB_ENV, "RUSTFLAGS=$flags$([Environment]::NewLine)")
Write-Host "Using RUSTFLAGS=$flags"
