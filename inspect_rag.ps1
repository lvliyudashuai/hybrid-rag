$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Output '=== FILES (top-level) ==='
Get-ChildItem $root | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Write-Output '=== chroma_db_qwen ==='
Get-ChildItem "$root\chroma_db_qwen" -ErrorAction SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize
Write-Output '=== README.md (first 80 lines) ==='
Get-Content "$root\README.md" -TotalCount 80
