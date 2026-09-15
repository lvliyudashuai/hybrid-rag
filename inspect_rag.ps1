$root = 'C:\Users\20992\Desktop\rag_demo'
Write-Output '=== FILES (top-level) ==='
Get-ChildItem $root | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Write-Output '=== chroma_db_qwen ==='
Get-ChildItem "$root\chroma_db_qwen" -ErrorAction SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize
Write-Output '=== README.md (first 80 lines) ==='
Get-Content "$root\README.md" -TotalCount 80
