# Download the Panda3D SDK (same version as the pinned wheel) that the native
# terrain collision solid compiles and links against on Windows: its headers,
# Eigen's included, and the import libraries the wheel lacks. Used by
# scripts/build_native.ps1 and CI. See docs/source/native_build.md.
#
#   scripts/fetch_native_deps.ps1 [-Dest <dir>]
#
# Dest: where to put it (default: .cache/native in the repository). Already
# fetched, it is kept. Needs 7-Zip. Prints the SDK directory, which holds
# include/ and lib/.
param([string]$Dest = "$PSScriptRoot/../.cache/native")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest crawls otherwise

# The version pyproject.toml pins (scripts/panda_version.sh).
$PandaVersion = (Select-String -Path "$PSScriptRoot/../pyproject.toml" `
    -Pattern '^Panda3D = "==(.*)"').Matches[0].Groups[1].Value
New-Item -ItemType Directory -Force $Dest | Out-Null
$Sdk = Join-Path (Resolve-Path $Dest) "panda3d-sdk-$PandaVersion-x64"

if (-not (Test-Path "$Sdk/include/pandaVersion.h")) {
    $SevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
    if (-not $SevenZip) { $SevenZip = "$env:ProgramFiles\7-Zip\7z.exe" }
    $Installer = Join-Path $env:TEMP "Panda3D-SDK-$PandaVersion-x64.exe"
    Invoke-WebRequest -OutFile $Installer `
        "https://www.panda3d.org/download/panda3d-$PandaVersion/Panda3D-SDK-$PandaVersion-x64.exe"
    # The installer is an NSIS archive: take the headers, and the import
    # libraries of the four DLLs the solid links against.
    & $SevenZip x -y -bso0 -bsp0 $Installer "-o$Sdk" "include/*" `
        lib/libpanda.lib lib/libpandaexpress.lib lib/libp3dtool.lib lib/libp3dtoolconfig.lib
    if ($LASTEXITCODE) { throw "7-Zip could not extract $Installer" }
    Remove-Item $Installer
}

$Sdk
