$seeds = 42, 123, 2026
$config = "configs/e2_early_fusion.yaml"

foreach ($seed in $seeds) {
    Write-Host "========================================"
    Write-Host "Running E2 (Early Fusion) with seed $seed"
    Write-Host "========================================"
    .venv\Scripts\python.exe -m scd.train --config $config --seed $seed
}
