# Chart Trainer

シンプルな株価チャート練習ツールです。Streamlitで動作し、`data/price_csv` に保存した株価データを読み込みます。

## 事前準備

1. Pythonパッケージをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
2. 本ツールは Python 3.8 で動作確認しています。仮想環境などで Python 3.8 を有効化してください。
   * ペアトレードの統計検定を利用する場合は `statsmodels` が必要です。`requirements.txt` に含まれているため、上記の手順で一緒に導入されます。
3. `.env` を作成し、J-Quantsのリフレッシュトークンを設定します。新仕様では開発者ポータルでクライアントID/シークレットを発行し、トークンを取得してから設定してください（ベースURLは未指定ならデフォルトを使用）。
   ```bash
   cp .env.example .env
   # .env を開いて値を設定してください
   ```

## 環境変数

- `JQUANTS_REFRESH_TOKEN`: 新仕様で発行したリフレッシュトークン（推奨・基本は必須）。開発者ポータルでクライアントID/シークレットを使って取得した値を設定します。
- `JQUANTS_BASE_URL`: J-Quants APIのベースURL（省略時 `https://api.jquants.com`）。
- `MAILADDRESS`: J-Quantsアカウントのメールアドレス（互換運用用。未設定時はリフレッシュトークンを使用）。
- `PASSWORD`: J-Quantsアカウントのパスワード（互換運用用）。
- `JQUANTS_CLIENT_ID`: 開発者ポータルのクライアントID（トークン取得時の控えとして `.env` に保持しても構いません）。
- `JQUANTS_CLIENT_SECRET`: 開発者ポータルのクライアントシークレット（同上）。
- `JQUANTS_TOKEN_SCOPE`: トークン発行時に指定したスコープ（例: `read`）。
- `EDINET_API_KEY`: EDINET APIキー（任意。未設定でも公開一覧の取得は可能ですが、設定を推奨）。

`app/config.py` で `data/db` と `data/price_csv` ディレクトリが自動で作成されます。

## データ取得（J-Quants）

サイドバーの「J-Quantsからダウンロード」ボタンから、銘柄コードと日付範囲を指定して最新データを取得できます。新仕様ではリフレッシュトークンの設定が前提です。トークンが未設定、日付の範囲が不正、またはAPIエラー時にはエラーメッセージが表示されます。

### プライム・スタンダード銘柄をまとめて更新する

`app/jquants_fetcher.py` には、プライム＋スタンダード（または listed_master.csv に含まれる全銘柄）のユニバースを対象に日足CSVを一括更新するCLIが用意されています。以下のように実行してください。

```bash
python -m app.jquants_fetcher
```

- 何もオプションを付けない場合は「プライム＋スタンダード」の銘柄を対象に、Freeプランで取得できる期間を増分更新します。
- 新仕様のリフレッシュトークンは `.env` の `JQUANTS_REFRESH_TOKEN` で設定します（互換運用として `MAILADDRESS`/`PASSWORD` も利用可能）。
- listed_master.csv に登録されている銘柄を市場区分問わず一括更新したい場合は `--use-listed-master` を付けてください。
- 毎回すべての取得可能期間を取り直したい場合は `--full-refresh` を付けます。
- 追加で取得したい銘柄がある場合は `data/meta/custom_symbols.txt` に1行1コードで記載し、`--include-custom` を付けて実行します（`--custom-path` で別ファイルを指定することも可能）。
- 特定の銘柄だけを更新したい場合は `--codes 7203 8306` のようにコードをスペース区切りで渡してください。
- 最新の株価を特定日付で追記したい場合は `--append-date 2024-12-30` のように日付を指定してください。日付のみ指定で全上場銘柄の株価を取得し、既存の銘柄ごとのCSVに当日の株価が追記されます。
- 立ち合い日 16:00 以降に当日日次データをユニバースへ自動反映したい場合は `--auto-after-close` を付けて実行してください。16:00 前または土日は自動でスキップされます（JST基準）。
- レートリミットに達した場合は Retry-After を尊重しつつ 5 分から最大 30 分まで指数的に待機しながら再試行します。夜通しで大量に取得する場合も、そのまま放置しておけば自動的にリトライされます。
- TOPIX 指数も同時に保存したい場合は `--include-topix` を付けてください。`data/price_csv/topix.csv` とメタ情報 `data/meta/topix.json` が作成・更新されます（取得期間はライトプランで取得可能な直近約5年分）。

実行すると `data/price_csv/{code}.csv` と `data/meta/{code}.json` が順次更新されます。API制限に配慮して銘柄ごとに簡易なウェイトが挿入されます。


### EDINETの開示情報を取得する

EDINETの提出書類一覧を `data/meta/edinet_disclosures.csv` に保存できます。

```bash
python -m app.jquants_fetcher --include-edinet --edinet-days 30
```

- `--edinet-days` で遡る日数を指定できます。
- 取得後、Streamlitの銘柄チャート画面で「適時開示（EDINET）」として表示されます。
TOPIX を含めて更新すると `data/price_csv/topix.csv` に日足ベースの OHLC が保存されます（コードは `TOPIX` 固定）。

### listed_master.csv の仕様変更について

最新仕様では上場銘柄マスタは `GET /v1/listed/info` で取得され、レスポンスの `listedInfo`（または `info`）にデータが入り、`pagination_key` があればページングされます。市場区分は `MarketCode`（例: `0111`）や `MarketCodeName`（例: `プライム`）で提供されるため、`listed_master.csv` の `market` 列は `PRIME`/`STANDARD`/`GROWTH` に正規化して保存します。あわせて `market_code` と `market_name` も保持します。

既存の `listed_master.csv` が旧仕様のままの場合は、以下のいずれかで再生成してください。

```bash
rm data/meta/listed_master.csv
python -m app.jquants_fetcher --use-listed-master
```

`listed_master.csv` が無い場合は自動的に再取得されます。市場区分の列が旧形式のままでも、可能な限り正規化してプライム＋スタンダードのユニバースを抽出します。

## アプリの起動

```bash
streamlit run ui_streamlit.py
```

ブラウザに表示されたページで銘柄と表示期間を選択してチャートを表示します。

## 手動分類（操作手順はこの1パターン）

手動でグループ分類を編集する場合は、**必ず「手動分類」タブ**を使用してください。サイドバーからは編集せず、以下の手順のみを案内対象とします。

1. 「手動分類」タブを開く。
2. 「編集対象グループ」で既存グループを選ぶか、「新規作成」を選んでグループ名を入力する。
3. 銘柄検索・チェック・一括追加・セクター反映などを使って銘柄を選択する。
4. 「保存/更新」で確定する（不要なグループは「削除」）。
5. 誤って追加した銘柄は、検索結果でチェックして「選択銘柄をマスタから取り消し」を押す。


## 動作確認（簡易チェック）

新仕様のリフレッシュトークンを設定した状態で、CLI経由の取得ができるか確認します。

```bash
# 例: トヨタ(7203)を直近営業日で追記（YYYY-MM-DDは直近営業日に置き換えてください）
python -m app.jquants_fetcher --codes 7203 --append-date YYYY-MM-DD
```

正常に終了すると `data/price_csv/7203.csv` と `data/meta/7203.json` が更新されます。Streamlit側でも同銘柄を選択して表示できることを確認してください。


## 強い上昇相場 × 最高値更新モメンタムのバックテスト

`app/backtest.py` に、TOPIX の Bull レジーム（200日移動平均上 + 200日線の上向き + 126日モメンタム正）で絞り込んだ
新高値更新モメンタム戦略の検証関数 `run_bull_market_new_high_momentum_backtest` を追加しています。

- 出来高・売買代金の大きい銘柄を優先するため、`top_liquidity_count` で流動性上位の銘柄数を指定できます。
- イベントスタディ（20日/60日の超過リターン）と、実運用想定の売買バックテスト（翌営業日寄りでエントリー、H日保有、コスト控除後）を同時に返します。
- 返却値は `signals / trades / daily_returns / event_study / summary` の辞書です。

```python
from app.backtest import run_bull_market_new_high_momentum_backtest

result = run_bull_market_new_high_momentum_backtest(
    symbols=None,                 # None なら data/price_csv 全銘柄
    high_lookback=252,            # 52週高値
    hold_days=20,                 # 保有日数
    event_cooldown_days=20,       # イベントの連発抑制
    top_liquidity_count=150,      # 流動性上位150銘柄
    one_way_cost_bps=15.0,        # 片道コスト(bp)
    rebalance_weekday=0,          # 月曜のみ新規建て（Noneで毎日）
)

print(result["summary"].to_string(index=False))
print(result["event_study"].to_string(index=False))
```

> 事前に `python -m app.jquants_fetcher --include-topix` で `data/price_csv/topix.csv` を更新しておくとスムーズです。

## 日米業種リードラグ（PCA SUB 論文再現）

`app/backtest.py` の `run_jp_us_sector_leadlag_backtest` と `ui_streamlit.py` の専用画面から、US の close-to-close をシグナル、JP の open-to-close をターゲットにした業種リードラグ戦略を実行できます。PCA を使って米国業種の共通要因を抽出し、日本側の寄り付き〜引けの相対的な強弱を検証する構成で、追加した操作はすべて Streamlit 画面上で完結します。

### 想定環境（PyCharm + Python 3.8 + Streamlit）

1. **PyCharm で Python 3.8 interpreter を選択**します。`File` → `Settings` → `Project: chart_trainer` → `Python Interpreter` から、Python 3.8 の仮想環境またはローカル環境を指定してください。
2. PyCharm の `Terminal` で依存関係を入れます。
   ```bash
   pip install -r requirements.txt
   ```
3. 同じく PyCharm の `Terminal`、または `Run/Debug Configurations` で以下を実行します。
   ```bash
   streamlit run ui_streamlit.py
   ```
4. ブラウザで開いた Streamlit 画面のバックテスト領域にある **「日米業種リードラグ（PCA SUB 論文再現）」** セクションから操作します。

> 将来この README に画面キャプチャを追加する場合は、上記 4. の直後に「バックテスト画面の全体像」や「CSVアップロード欄」の画像を差し込むと、初回セットアップから実行導線までを自然につなげられます。

### 論文戦略の概要

- 米国 ETF 群の日次リターン（close-to-close）を説明変数として扱います。
- 日本 ETF 群の日中リターン（open-to-close）を被説明変数として扱います。
- 事前期間（prior期間）で基準関係を推定し、本番期間で long / short シグナルとウェイトを計算します。
- `baseline_mode` の既定値は `pca_sub` で、README では論文再現の基本モードとしてこの設定を前提に案内します。

### 既定値

画面上の既定値は次のとおりです。

- `lookback=60`
- `lambda_reg=0.9`
- `n_components=3`
- `quantile_q=0.3`

必要に応じて `one_way_cost_bps`、`期間`、`prior期間` も画面上で変更できますが、まずは既定値のまま動作確認するのが安全です。

### 必要データ

- **US 業種 ETF の CSV**
- **Japan 業種 ETF の CSV**
- 各 CSV は、少なくとも日付と価格系列を含み、Streamlit 画面から複数ファイルをまとめて保存できる形にしておきます。
- 事前期間（prior期間）と本番期間（期間）の両方をカバーできる履歴長を用意してください。

### US / Japan ETF CSV の保存先

Streamlit 画面のアップロード機能で保存されるディレクトリは以下です。

- US ETF CSV: `data/price_csv/leadlag_us/`
- Japan ETF CSV: `data/price_csv/leadlag_jp/`

既存運用でファイルを手動配置する場合も、この保存先に置いておくと画面操作と整合します。

### CSVアップロード手順

1. Streamlit 画面で **「日米業種リードラグ（PCA SUB 論文再現）」** を開きます。
2. US 側は 2 通りの導線があります（どちらも同じ画面内で完結します）。
   - **画面から直接取得する場合（推奨）**:  
     `USデータ取得（画面から直接取得）` で **USティッカー**、**US取得期間**、**取得元** を指定し、**「USデータ取得して保存」** を押します。`data/price_csv/leadlag_us/` に保存され、直下にプレビューが表示されます。
   - **手元CSVを使う場合**:  
     `US CSV（保存先: data/price_csv/leadlag_us/）` で複数CSVを選び、**「US CSVを保存」** を押します。
3. JP 側で `JP CSV（保存先: data/price_csv/leadlag_jp/）` を選び、**「JP CSVを保存」** を押します。
4. 保存後は同じ画面のままバックテスト実行へ進めます。別ツールへ移動する必要はありません。

### 直近欠損の扱い

本アプリでは、lead-lag セクションに **「データ健全性チェック」** を追加し、US/JP 各銘柄ごとに `latest_date`、`stale_days`、`missing_recent_dates`、`missing_value_rows`、`is_usable` などを一覧表示します。チェック対象は画面から保存した CSV で、追加操作なしでそのまま同画面内で確認できます。

- **祝日ズレと単純欠落の違い**  
  日米で祝日が異なるため、完全一致の日付列を前提にせず、市場（`market`）ごとに営業日を計算する構造で判定します。現時点の実装は「観測レンジ内の平日ベース営業日」を期待日として使い、将来は祝日カレンダー実装に置き換え可能です。
- **警告表示の意味**  
  `stale_days` は「最新データ日が直近営業日から何日遅れているか」、`missing_recent_dates` は「直近 N 営業日で欠けている日付」、`missing_recent_ratio` はその欠落率です。これに重複日、OHLCV 欠損行、連続欠損の有無を加味して `is_usable` を判定します。
- **除外ポリシー**  
  バックテスト前に次の 3 ポリシーを画面で選べます。  
  1. **欠損銘柄を除外して続行**: `is_usable=False` の銘柄を除外して実行  
  2. **警告のみ**: 警告表示は出すが全銘柄で実行  
  3. **欠損があれば停止**: 1 銘柄でも `is_usable=False` があれば実行を中止

PyCharm + Python 3.8 環境でも、上記チェックとポリシー選択はすべて Streamlit 画面上だけで完結します。

### Streamlit画面での実行手順

1. `lookback`、`lambda_reg`、`n_components`、`quantile_q` を確認します。初回は既定値のまま推奨です。
2. `期間` に本番バックテスト期間を設定します。
3. `prior期間` に事前学習・基準推定用の期間を設定します。
4. 必要に応じて `one_way_cost_bps` と `baseline_mode` を調整します。論文再現の入口としては `baseline_mode=pca_sub` を推奨します。
5. **「バックテスト実行」** を押します。
6. 実行後、同一画面内で `summary`、`daily_returns`、`signals`、`weights`、`diagnostics` が順番に表示されます。

### 検索・フィルタ方法

バックテスト実行後は、画面上で次の絞り込みが可能です。

- `コード検索`: ETF コードで対象を絞り込み
- `long / short`: 売買方向で絞り込み
- `セクター名検索`: 業種・グループ名で絞り込み
- `baseline 切替`: `pca_sub` などの方式別に確認
- `日付範囲`: 表示対象期間を再指定
- `表示件数上限`: 表示する行数を制御

`signals` と `weights` は同じ検索条件で見比べられるため、シグナル発生と配分の対応確認を画面だけで進められます。

### CSVダウンロード方法

各結果テーブルの下にダウンロードボタンがあります。必要な表だけをその場で CSV 保存できます。

- `daily_returns CSVをダウンロード`
- `signals CSVをダウンロード`
- `weights CSVをダウンロード`
- `diagnostics(日次) CSVをダウンロード`

補助的に `mapping_df` と `trades_df` も画面から CSV 保存できます。再分析や Excel / pandas での二次確認に便利です。

### 返却される出力

Bull breakout セクションと同様に、実行結果として主要な出力を確認できます。

- `summary`: CAGR、Sharpe、最大ドローダウン、勝率、件数などの集計サマリー
- `daily_returns`: 日次の戦略損益、ベンチマーク差分、`equity_curve` を含む時系列
- `signals`: いつ、どの業種・銘柄群に long / short シグナルが出たかを確認する一覧
- `weights`: 各日付・各サイドの配分ウェイトを確認する一覧
- `diagnostics`: 件数集計、日次ログ、正則化や設定情報などの診断用出力

画面上では `summary` を表で確認しつつ、`daily_returns` はテーブルとエクイティカーブ、`signals` / `weights` / `diagnostics` は表 + CSV ダウンロードで確認する流れになります。

### オーバーフィット注意

- `lookback`、`n_components`、`quantile_q` を何度も微調整して、特定期間だけ良く見える設定を選ぶと過学習になりやすいです。
- `prior期間` と `期間` を明確に分け、評価期間を後から触りすぎない運用を推奨します。
- 1回の好成績だけで判断せず、複数期間・複数相場環境で再現性を確認してください。
- CSV を追加・削除するとユニバース自体が変わるため、結果比較時は使用データセットも合わせて記録してください。

## ペアトレードのバックテスト

`app/pair_trading.py` に簡易のペアトレードバックテストを追加しています。業種（`sector33` / `sector17`）ごとにペア候補を生成し、ローリングOLSでヘッジ比率を推定してスプレッドのZスコアを用いて売買します。

```bash
python - <<'PY'
from app.pair_trading import (
    PairTradeConfig,
    backtest_pairs,
    generate_pairs_by_sector,
    optimize_pair_trade_parameters,
)

config = PairTradeConfig(lookback=60, entry_z=2.0, exit_z=0.5, max_holding_days=20)
pairs = generate_pairs_by_sector(sector_col="sector33", max_pairs_per_sector=10)
trades_df, summary_df = backtest_pairs(pairs, config=config)

print(summary_df.head(10).to_string(index=False))
print(trades_df.head(10).to_string(index=False))

param_grid = {
    "lookback": [40, 60, 80],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_z": [0.3, 0.5],
    "stop_z": [3.0, 3.5],
    "max_holding_days": [10, 20],
}
eval_df = optimize_pair_trade_parameters(pairs, param_grid=param_grid, min_trades=5)
print(eval_df.head(10).to_string(index=False))
PY
```

## AIエージェントを投資に活用するための最小プロトタイプ（Python 3.8 / PyCharm）

いきなり自動売買を行うのではなく、まずは「調査・分析・リスク確認を自動化し、最終判断は人間が行う」構成から始めるのがおすすめです。

### 1週間で作る最小構成

以下のファイルを追加するだけで、日次運用の型を作れます。

- `agent/data_collector.py`: 株価データ・ニュースの取得
- `agent/signal_model.py`: シンプルなシグナル算出（例: モメンタム + トレンド）
- `agent/risk_guard.py`: 最大許容損失、ポジション偏り、ボラティリティチェック
- `agent/reporter.py`: 提案内容と根拠を日次レポートとして出力
- `agent/main.py`: 上記を順番に実行するエントリーポイント

PyCharmでは `agent/main.py` の Run Configuration を作り、まずは手動実行で回すとデバッグしやすくなります。

### 推奨ワークフロー

1. **データ収集**: 銘柄ユニバースを決め、終値・出来高・指標を日次で更新
2. **シグナル生成**: ルールベースで候補銘柄を抽出
3. **リスク判定**: 1トレード損失上限・日次損失上限・銘柄集中をチェック
4. **レポート出力**: 「提案・根拠・リスク」をテキストで残す
5. **人間が最終判断**: 注文は手動で実施（最初は必須）

### 最低限の安全ルール

- 1トレードあたり損失上限を8% に制限
- 取引ログ（提案時刻、根拠、実行有無、損益）を必ず保存
- モデル更新は場中ではなく、週次や月次など定期バッチに固定

### 検証方法（実資金投入前）

- 最低3か月はペーパートレード
- 次の指標を必ず確認
  - 累積リターン
  - 最大ドローダウン
  - 勝率
  - Sharpe比
  - 手数料・スリッページ考慮後の成績

このプロジェクトの既存機能（CSV更新、バックテスト）と組み合わせることで、AIエージェントを無理なく段階導入できます。
