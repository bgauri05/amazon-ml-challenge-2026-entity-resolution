$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $project
$python = (Get-Command python).Source
$status = Join-Path $project 'student_resource\output_v10\run_status.json'
try {
    @{ state = 'inference_running'; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
    $run = Start-Process -FilePath $python -ArgumentList @('-B','-u','model_v10_bounded.py','--mode','test','--k','2','--batch','400') -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $project 'v10.log') -RedirectStandardError (Join-Path $project 'v10.stderr.log')
    if ($run.ExitCode -ne 0) { throw "V10 inference failed with exit code $($run.ExitCode); inspect v10.stderr.log" }
    @{ state = 'validating'; updated = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
    $check = Start-Process -FilePath $python -ArgumentList @('-B','-u','student_resource/utils/validate_submission.py','--matching','student_resource/output_v10/matching_results.tsv','--candidate','student_resource/output_v10/candidate_pairs.tsv','--test-dir','student_resource/dataset/test','--check-ids') -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $project 'validation_v10.log') -RedirectStandardError (Join-Path $project 'validation_v10.stderr.log')
    if ($check.ExitCode -ne 0) { throw "Official validator failed with exit code $($check.ExitCode); inspect validation_v10.log" }
    @{ state = 'ready'; updated = (Get-Date).ToString('o'); matching = 'student_resource/output_v10/matching_results.tsv'; validation = 'PASS with ID checks' } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
} catch {
    @{ state = 'failed'; updated = (Get-Date).ToString('o'); error = $_.Exception.Message } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
    throw
}
