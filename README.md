# Chart Trainer

シンプルな株価チャート練習ツールです。Streamlitで動作し、`data/price_csv` に保存した株価データを読み込みます。

## 事前準備

1. Pythonパッケージをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
2. `.env` を作成し、J-QuantsのリフレッシュトークンとベースURLを設定します（ベースURLは未指定ならデフォルトを使用）。
   ```bash
   cp .env.example .env
   # .env を開いて値を設定してください
   ```

## 環境変数

- `JQUANTS_REFRESH_TOKEN`: J-Quantsから取得したリフレッシュトークン。
- `JQUANTS_BASE_URL`: J-Quants APIのベースURL（省略時 `https://api.jquants.com`）。

`app/config.py` で `data/db` と `data/price_csv` ディレクトリが自動で作成されます。

## データ取得（J-Quants）

サイドバーの「J-Quantsからダウンロード」ボタンから、銘柄コードと日付範囲を指定して最新データを取得できます。トークンが未設定、日付の範囲が不正、またはAPIエラー時にはエラーメッセージが表示されます。

## アプリの起動

```bash
streamlit run ui_streamlit.py
```

ブラウザに表示されたページで銘柄と表示期間を選択してチャートを表示します。
