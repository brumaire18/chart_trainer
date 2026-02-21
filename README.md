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
