param(
    [string]$SiteRoot = "docs",
    [string]$PairSourceRoot = "",
    [switch]$SkipPairDataRefresh,
    [switch]$RefreshFundRemote,
    [switch]$IncludeFundDisclosures
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path $PSScriptRoot
Set-Location $repoRoot

$siteDir = if ([System.IO.Path]::IsPathRooted($SiteRoot)) { $SiteRoot } else { Join-Path $repoRoot $SiteRoot }
$pairDir = Join-Path $siteDir "pair"
$fundDir = Join-Path $siteDir "fund"

$docsRoot = [Environment]::GetFolderPath("MyDocuments")
if ([string]::IsNullOrWhiteSpace($PairSourceRoot)) {
    $PairSourceRoot = Join-Path $docsRoot "chart_trainer_pair_reports"
}

New-Item -ItemType Directory -Force -Path $siteDir | Out-Null
New-Item -ItemType Directory -Force -Path $pairDir | Out-Null
New-Item -ItemType Directory -Force -Path $fundDir | Out-Null

$pairArgs = @{
    OutputRoot = $PairSourceRoot
    PublishRoot = $pairDir
}
if ($SkipPairDataRefresh.IsPresent) {
    $pairArgs["SkipRefresh"] = $true
}
& (Join-Path $repoRoot "run_morning_pair_recommendation_task.ps1") @pairArgs

$pythonExe = Join-Path $repoRoot ".conda38\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python runtime not found: $pythonExe"
}
$sitePackages = Resolve-Path (Join-Path $repoRoot "Lib\site-packages")
$env:PYTHONHOME = Join-Path $repoRoot ".conda38"
$env:PYTHONPATH = ""

$pairPublishBootstrap = @"
import runpy
import sys
sys.path.insert(0, r'$($sitePackages.Path)')
sys.argv = [
    'scripts/publish_pair_trading_snapshot.py',
    '--source-root', r'$PairSourceRoot',
    '--publish-root', r'$pairDir',
    '--title', 'Pair Trading Daily Dashboard',
]
runpy.run_path('scripts/publish_pair_trading_snapshot.py', run_name='__main__')
"@
& $pythonExe -c $pairPublishBootstrap

$fundArgs = @{
    PublishRoot = $fundDir
    Title = "Pseudo Fund Dashboard"
}
if ($RefreshFundRemote.IsPresent) {
    $fundArgs["RefreshRemote"] = $true
}
if ($IncludeFundDisclosures.IsPresent) {
    $fundArgs["IncludeDisclosures"] = $true
}
& (Join-Path $repoRoot "publish_fund_system_snapshot.ps1") @fundArgs

$publishedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss K")
$indexHtml = @"
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chart Trainer Mobile Dashboards</title>
  <style>
    :root {
      --bg: #f4efe6;
      --ink: #1d2733;
      --muted: #5d6b79;
      --card: rgba(255,255,255,0.94);
      --line: #d7e0e7;
      --accent: #0f766e;
      --accent-2: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Hiragino Sans", "Yu Gothic", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(15,118,110,0.16), transparent 28%),
        radial-gradient(circle at bottom left, rgba(180,83,9,0.14), transparent 26%),
        linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 18px;
    }
    .hero {
      border-radius: 24px;
      padding: 24px;
      color: #fff;
      background: linear-gradient(135deg, #1f2937, #0f766e 64%, #d97706);
      box-shadow: 0 18px 42px rgba(15, 23, 42, 0.16);
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 7vw, 44px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 8px 0 0;
      font-size: 15px;
      line-height: 1.6;
      max-width: 58ch;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }
    .card {
      display: block;
      text-decoration: none;
      color: inherit;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    }
    .eyebrow {
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      background: #ecfeff;
      color: var(--accent);
      margin-bottom: 12px;
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
    }
    .meta {
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
    }
    .footer {
      margin: 18px 0 8px;
      font-size: 13px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Chart Trainer Dashboards</h1>
      <p>スマートフォンから見やすい公開ページです。ペアトレーディング画面とファンド全体画面を分けて置いています。</p>
      <p class="meta">Published at: $publishedAt</p>
    </section>

    <section class="grid">
      <a class="card" href="./pair/">
        <span class="eyebrow">PAIR</span>
        <h2>Pair Trading</h2>
        <p>ペア候補、ポートフォリオシミュレーション、改善サイクルの結果を確認できます。</p>
      </a>
      <a class="card" href="./fund/">
        <span class="eyebrow">FUND</span>
        <h2>Pseudo Fund</h2>
        <p>保有、現金、リスク警告、戦略候補、日次レポートをまとめて確認できます。</p>
      </a>
    </section>

    <p class="footer">GitHub Pages 用の静的サイトです。OneDrive アプリではなく、通常のブラウザで表示する前提です。</p>
  </main>
</body>
</html>
"@

Set-Content -Path (Join-Path $siteDir "index.html") -Value $indexHtml -Encoding UTF8
Set-Content -Path (Join-Path $siteDir "404.html") -Value $indexHtml -Encoding UTF8
