param([string]$DataPath = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($DataPath -eq '') {
  python src/oil_lstm.py --forecast-only
} else {
  python src/oil_lstm.py --forecast-only --data $DataPath
}
if ($LASTEXITCODE -ne 0) { throw 'Frozen-checkpoint forecast refresh failed.' }
