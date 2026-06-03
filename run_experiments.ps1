# Define the parameter ranges
$workers = 1..5
$betas = @(0.5, 1, 2, 5)

# Calculate 6 log-separated values from 0.0005 to 0.05
$eta_start = 0.0005
$eta_end = 0.05
$eta_count = 4

$log_start = [Math]::Log10($eta_start)
$log_end = [Math]::Log10($eta_end)
$log_step = ($log_end - $log_start) / ($eta_count - 1)

$etas = @()
for ($i = 0; $i -lt $eta_count; $i++) {
    $exponent = $log_start + ($i * $log_step)
    $val = [Math]::Round([Math]::Pow(10, $exponent), 6)
    $etas += $val
}

# Generate exactly 5 values from 0.1 to 1.0 for alpha and alpha_pull
$alphas = @()
$alpha_pulls = @()
for ($i = 0; $i -le 4; $i++) {
    # 0.1 + (i * 0.225) spaces 5 values perfectly from 0.1 to 1.0
    $val = [Math]::Round(0.1 + ($i * 0.225), 4)
    $alphas += $val
    $alpha_pulls += $val
}

# Total experiment counter
$totalRuns = $workers.Count * $etas.Count * $betas.Count * $alphas.Count * $alpha_pulls.Count
$currentRun = 1

Write-Host "Starting grid search with log-spaced etas: ($($etas -join ', '))" -ForegroundColor Cyan
Write-Host "Alpha values: ($($alphas -join ', '))" -ForegroundColor Cyan
Write-Host "Total scheduled runs: $totalRuns" -ForegroundColor Cyan
Write-Host "--------------------------------------------------"

# Nested loops to iterate through all combinations
foreach ($w in $workers) {
    foreach ($e in $etas) {
        foreach ($b in $betas) {
            foreach ($a in $alphas) {
                foreach ($ap in $alpha_pulls) {
                    
                    Write-Host "[Run $currentRun/$totalRuns] Executing with: --num_workers $w --eta $e --beta $b --alpha $a --alpha_pull $ap" -ForegroundColor Yellow
                    
                    # Run the python command
                    py run_optimization.py --optimizer easgd2 --num_epochs 2000 --num_workers $w --eta $e --beta $b --alpha $a --alpha_pull $ap
                    
                    $currentRun++
                }
            }
        }
    }
}

Write-Host "All optimization runs completed!" -ForegroundColor Green