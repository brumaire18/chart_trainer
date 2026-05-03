param(
    [string]$PairPublishRoot = "",
    [string]$FundPublishRoot = "",
    [string]$PagesSiteRoot = "",
    [switch]$SkipPairDataRefresh,
    [switch]$RefreshFundRemote,
    [switch]$IncludeFundDisclosures,
    [switch]$PublishGitHubPages
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path $PSScriptRoot
Set-Location $repoRoot

$docsRoot = [Environment]::GetFolderPath("MyDocuments")
if ([string]::IsNullOrWhiteSpace($PairPublishRoot)) {
    $PairPublishRoot = Join-Path $docsRoot "chart_trainer_pair_reports"
}
if ([string]::IsNullOrWhiteSpace($FundPublishRoot)) {
    $FundPublishRoot = Join-Path $docsRoot "chart_trainer_fund_dashboard"
}
if ([string]::IsNullOrWhiteSpace($PagesSiteRoot)) {
    $PagesSiteRoot = Join-Path $repoRoot "docs"
}

$runtimeDir = Join-Path $repoRoot "runtime\mobile_dashboards"
if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}
$logPath = Join-Path $runtimeDir "mobile_dashboards_refresh.log"
$statusPath = Join-Path $runtimeDir "latest_mobile_dashboards_status.json"

Add-Content -Path $logPath -Value ("[{0}] refresh start" -f (Get-Date).ToString("s"))

try {
    $pairArgs = @{
        OutputRoot = $PairPublishRoot
        PublishRoot = $PairPublishRoot
    }
    if ($SkipPairDataRefresh.IsPresent) {
        $pairArgs["SkipRefresh"] = $true
    }
    & (Join-Path $repoRoot "run_morning_pair_recommendation_task.ps1") @pairArgs

    $fundArgs = @{
        PublishRoot = $FundPublishRoot
        Title = "Pseudo Fund Dashboard"
    }
    if ($RefreshFundRemote.IsPresent) {
        $fundArgs["RefreshRemote"] = $true
    }
    if ($IncludeFundDisclosures.IsPresent) {
        $fundArgs["IncludeDisclosures"] = $true
    }
    & (Join-Path $repoRoot "publish_fund_system_snapshot.ps1") @fundArgs

    if ($PublishGitHubPages.IsPresent) {
        $pagesArgs = @{
            SiteRoot = $PagesSiteRoot
        }
        if ($SkipPairDataRefresh.IsPresent) {
            $pagesArgs["SkipPairDataRefresh"] = $true
        }
        if ($RefreshFundRemote.IsPresent) {
            $pagesArgs["RefreshFundRemote"] = $true
        }
        if ($IncludeFundDisclosures.IsPresent) {
            $pagesArgs["IncludeFundDisclosures"] = $true
        }
        & (Join-Path $repoRoot "publish_github_pages_site.ps1") @pagesArgs
    }

    $pairIndexPath = Join-Path $PairPublishRoot "index.html"
    $fundIndexPath = Join-Path $FundPublishRoot "index.html"
    $pagesIndexPath = Join-Path $PagesSiteRoot "index.html"
    $status = [ordered]@{
        refreshedAt = (Get-Date).ToString("o")
        pairPublishRoot = $PairPublishRoot
        fundPublishRoot = $FundPublishRoot
        pagesSiteRoot = $PagesSiteRoot
        pairIndexUpdatedAt = if (Test-Path $pairIndexPath) { (Get-Item $pairIndexPath).LastWriteTime.ToString("o") } else { $null }
        fundIndexUpdatedAt = if (Test-Path $fundIndexPath) { (Get-Item $fundIndexPath).LastWriteTime.ToString("o") } else { $null }
        pagesIndexUpdatedAt = if (Test-Path $pagesIndexPath) { (Get-Item $pagesIndexPath).LastWriteTime.ToString("o") } else { $null }
        success = $true
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content -Path $statusPath -Encoding UTF8
    Add-Content -Path $logPath -Value ("[{0}] refresh success" -f (Get-Date).ToString("s"))
}
catch {
    $status = [ordered]@{
        refreshedAt = (Get-Date).ToString("o")
        pairPublishRoot = $PairPublishRoot
        fundPublishRoot = $FundPublishRoot
        success = $false
        error = $_.Exception.Message
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content -Path $statusPath -Encoding UTF8
    Add-Content -Path $logPath -Value ("[{0}] refresh failed: {1}" -f (Get-Date).ToString("s"), $_.Exception.Message)
    throw
}
