# verify.ps1 - quick leak check after patching (Windows PowerShell).
# Run from repo root:  powershell -ExecutionPolicy Bypass -File verify.ps1
#
# NOTE: This checks the ASCII-detectable leaks (titles, refs, names, URLs).
# For the full Vietnamese-diacritic scan, run  bash verify.sh  in Git Bash
# (Git for Windows ships Git Bash), which is the authoritative check.

if (-not (Test-Path "run_main.py")) {
    Write-Host "ERROR: run from repo root (where run_main.py lives)." -ForegroundColor Red
    exit 1
}

$exts  = @("*.py","*.md","*.yaml","*.yml","*.txt")
$files = Get-ChildItem -Recurse -Include $exts -File |
         Where-Object { $_.FullName -notmatch '\\(outputs|logs|\.git|\.venv)\\' }

$fail = $false

function Scan($label, $pattern) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    $hits = $files | Select-String -Pattern $pattern
    if ($hits) {
        $hits | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Yellow }
        $script:fail = $true
    } else {
        Write-Host "  clean" -ForegroundColor Green
    }
    Write-Host ""
}

Scan "Internal refs (main_vi / ablation_vi / generalrule)"            'main_vi|ablation_vi|[Gg]eneralrule'
Scan "Project/venue leaks (Paper 2 / MAPR / thesis / sister repos)"   'Paper 2|paper2| P2 |MAPR|thesis|defense|brfss-diabetes|diabetes-xai-agreement'
Scan "Stale title / clone URL / date / personal email"               'Constraint Compiler|git clone https://github.com/thieuanhvan|01/06/2026|target submission|thieuanhvan@gmail'

# P4 label: expect ONLY the 2 figure-legend lines
Write-Host "=== Standalone P4 label (expect only 2 figure legends) ===" -ForegroundColor Cyan
$p4 = $files | Select-String -Pattern '\bP4\b'
if ($p4) {
    $p4 | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) }
    $bad = $p4 | Where-Object { $_.Line -notmatch 'directional taxonomy' }
    if ($bad) { Write-Host "  WARNING: P4 beyond the 2 figure legends." -ForegroundColor Yellow; $fail = $true }
    else { Write-Host "  OK - only the 2 intentional figure-legend labels." -ForegroundColor Green }
} else { Write-Host "  clean (no P4 at all)" -ForegroundColor Green }
Write-Host ""

if (-not $fail) {
    Write-Host "RESULT: ASCII checks CLEAN. Run 'bash verify.sh' in Git Bash for the full Vietnamese scan." -ForegroundColor Green
    exit 0
} else {
    Write-Host "RESULT: issues remain (see above)." -ForegroundColor Red
    exit 1
}
