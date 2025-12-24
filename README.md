# Chart Trainer

シンプルな株価チャート練習ツールです。Streamlitで動作し、`data/price_csv` に保存した株価データを読み込みます。

## 事前準備

1. Pythonパッケージをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
2. `.env` を作成し、J-Quantsのメールアドレス・パスワード・ベースURL、（必要ならリフレッシュトークン）を設定します（ベースURLは未指定ならデフォルトを使用）。
   ```bash
   cp .env.example .env
   # .env を開いて値を設定してください
   ```

## 環境変数

- `JQUANTS_REFRESH_TOKEN`: J-Quantsから取得したリフレッシュトークン（未指定の場合、`MAILADDRESS` と `PASSWORD` から毎回取得します）。
- `JQUANTS_BASE_URL`: J-Quants APIのベースURL（省略時 `https://api.jquants.com`）。
- `MAILADDRESS`: J-Quantsアカウントのメールアドレス。
- `PASSWORD`: J-Quantsアカウントのパスワード。

`app/config.py` で `data/db` と `data/price_csv` ディレクトリが自動で作成されます。

## データ取得（J-Quants）

サイドバーの「J-Quantsからダウンロード」ボタンから、銘柄コードと日付範囲を指定して最新データを取得できます。トークンが未設定、日付の範囲が不正、またはAPIエラー時にはエラーメッセージが表示されます。

### プライム・スタンダード銘柄をまとめて更新する

`app/jquants_fetcher.py` には、プライム＋スタンダード（または listed_master.csv に含まれる全銘柄）のユニバースを対象に日足CSVを一括更新するCLIが用意されています。以下のように実行してください。

```bash
python -m app.jquants_fetcher
```

- 何もオプションを付けない場合は「プライム＋スタンダード」の銘柄を対象に、Freeプランで取得できる期間を増分更新します。
- listed_master.csv に登録されている銘柄を市場区分問わず一括更新したい場合は `--use-listed-master` を付けてください。
- 毎回すべての取得可能期間を取り直したい場合は `--full-refresh` を付けます。
- 追加で取得したい銘柄がある場合は `data/meta/custom_symbols.txt` に1行1コードで記載し、`--include-custom` を付けて実行します（`--custom-path` で別ファイルを指定することも可能）。
- 特定の銘柄だけを更新したい場合は `--codes 7203 8306` のようにコードをスペース区切りで渡してください。
- 最新の株価を特定日付で追記したい場合は `--append-date 2024-12-30` のように日付を指定してください。日付のみ指定で全上場銘柄の株価を取得し、既存の銘柄ごとのCSVに当日の株価が追記されます。
- レートリミットに達した場合は Retry-After を尊重しつつ 5 分から最大 30 分まで指数的に待機しながら再試行します。夜通しで大量に取得する場合も、そのまま放置しておけば自動的にリトライされます。
- TOPIX 指数も同時に保存したい場合は `--include-topix` を付けてください。`data/price_csv/topix.csv` とメタ情報 `data/meta/topix.json` が作成・更新されます（取得期間はライトプランで取得可能な直近約5年分）。

実行すると `data/price_csv/{code}.csv` と `data/meta/{code}.json` が順次更新されます。API制限に配慮して銘柄ごとに簡易なウェイトが挿入されます。
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
