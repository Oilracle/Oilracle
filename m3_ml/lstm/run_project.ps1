$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -u src/oil_lstm.py
if ($LASTEXITCODE -ne 0) { throw 'Model run failed.' }
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
Write-Host 'Completed. Results are in outputs/frozen_october_2026/.'
