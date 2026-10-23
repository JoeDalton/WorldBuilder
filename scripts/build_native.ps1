# Build the native terrain collision solid (Windows x64) for the installed
# panda3d wheel, with MSVC. See docs/source/native_build.md.
#
#   scripts/build_native.ps1 [-Python <python>] [-Sdk <panda3d sdk>]
#
# Python: the interpreter whose panda3d wheel to build for (default: python).
# Sdk: a Panda3D SDK of the same version, holding include/ and lib/ (default:
# fetched by scripts/fetch_native_deps.ps1 into .cache/native). MSVC is found
# with vswhere when cl isn't on PATH.
# Writes src/world_builder/terrain/native/libterrain_collision-panda<major>.<minor>.dll
param([string]$Python = "python", [string]$Sdk)
$ErrorActionPreference = "Stop"

if (-not $Sdk) { $Sdk = & "$PSScriptRoot/fetch_native_deps.ps1" }

$WheelVersion = & $Python -c "import panda3d; print(panda3d.__version__)"
if ($LASTEXITCODE) { throw "$Python has no panda3d" }
$Series = ($WheelVersion -split '\.')[0..1] -join '.'
$Headers = (Select-String -Path "$Sdk/include/pandaVersion.h" `
    -Pattern '#define PANDA_VERSION_STR "(.*)"').Matches[0].Groups[1].Value
if ($Headers -ne $WheelVersion) {
    throw "headers are Panda3D $Headers, the wheel is ${WheelVersion}: use headers of the same version"
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $Vs = & $VsWhere -latest -products * -property installationPath `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
    if (-not $Vs) { throw "no Visual Studio with the x64 C++ tools" }
    cmd /c "`"$Vs\VC\Auxiliary\Build\vcvars64.bat`" > nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
        }
    }
}

$Src = (Resolve-Path "$PSScriptRoot/../src/world_builder/terrain/native").Path
$Out = Join-Path $Src "libterrain_collision-panda$Series.dll"
$Work = (New-Item -ItemType Directory -Force (Join-Path $env:TEMP "wb_native")).FullName

# /MD: the C++ runtime of Panda's DLLs and Python. Run from a temporary
# directory, which gets the object file; the import library goes there too.
# Eigen's headers come with the SDK's.
Push-Location $Work
try {
    & cl.exe /nologo /std:c++14 /O2 /EHsc /MD /utf-8 /DNDEBUG /LD `
        "/I$Sdk\include" "$Src\terrain_collision.cpp" "/Fe$Out" `
        /link "/LIBPATH:$Sdk\lib" "/IMPLIB:$Work\terrain_collision.lib" `
        libpanda.lib libpandaexpress.lib libp3dtool.lib libp3dtoolconfig.lib
    if ($LASTEXITCODE) { throw "cl failed" }
    & dumpbin.exe /nologo /dependents $Out
} finally {
    Pop-Location
}
"built $Out"
