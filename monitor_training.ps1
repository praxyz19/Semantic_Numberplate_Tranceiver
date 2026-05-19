# Real-time training progress monitor
$historyPath = "artifacts\history.json"
$lastCount = 0

while ($true) {
    if (Test-Path $historyPath) {
        $history = Get-Content $historyPath | ConvertFrom-Json
        $current = $history.Count
        
        if ($current -gt $lastCount) {
            $latest = $history[-1]
            $round = $latest.round
            $update = $latest.server_update
            $client = $latest.client_id
            $loss = [math]::Round($latest.loss, 3)
            $text_loss = [math]::Round($latest.text_loss, 3)
            $recon_loss = [math]::Round($latest.recon_loss, 4)
            
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Round $round | Update $update | Client $client | Loss: $loss | Text: $text_loss | Recon: $recon_loss" -ForegroundColor Green
            $lastCount = $current
        }
    }
    Start-Sleep -Seconds 15
}
