# llm-as-a-judge-sample

このリポジトリは、Zenn の記事「[LLM-as-a-Judge とルーブリック評価](https://zenn.dev/ubie_dev/articles/llm-as-a-judge-rubric-evaluation)」のサンプルコードです。

記事内で紹介している 3 つの評価手法（主観評価、具体的な基準を用いた評価、ルーブリック評価）を実際に試すことができます。

## プロジェクト構成

```text
.
├── evaluation.ipynb       # 評価実行用の Notebook
├── generation.ipynb       # 評価対象のデータ生成用の Notebook
├── src/
│   ├── data/             # データセットおよび評価結果の保存先
│   ├── evaluator/        # 評価ロジックとプロンプト
│   └── generator/        # 回答生成ロジック
└── pyproject.toml        # 依存関係定義
```

## セットアップ

このプロジェクトは [uv](https://github.com/astral-sh/uv) を使用してパッケージ管理を行っています。

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. 環境変数の設定

`src/models.py` に適当な API Key などを指定してください。

本サンプルでは、評価に Google Gemini、データ生成に Anthropic Claude を使用する構成になっています。

## 使い方

### 1. データの生成 (Optional)

`generation.ipynb` を実行すると、設定されたシナリオに基づいて LLM (デフォルトでは Claude) が回答を生成し、`src/data/evaluation_dataset.json` に保存します。
※ すでにデータセットが存在する場合はスキップ可能です。

### 2. 評価の実行

`evaluation.ipynb` を実行すると、以下の 3 つの評価手法が順に実行されます。

1. **Subjective Evaluation**: 抽象的な基準（1〜5点）による評価
2. **General Evaluation**: 具体的な基準（1〜5点）による評価
3. **Rubric Evaluation**: ルーブリック（True/False）による評価

評価結果は `src/data/evaluation_result/YYYY-MM-DD-HH-MM-SS/` ディレクトリ配下に JSON 形式で保存されます。

デフォルトでは評価のばらつきを確認するために 50 回試行する設定になっています。試行回数は Notebook 内の `ITERATION_COUNT` 変数で変更可能です。
