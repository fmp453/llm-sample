# LLMの実験

## ファイルの内容
- `requirements.txt`: pipでインストールするもの
- `mmlu.py`: MMLUのサイバーセキュリティ分野の評価をするもの
- `gpt2-finetune.py`: GPT-2をfine-tuningする際のサンプルコード (そのままではデータセットがないため実行不可)
- `peft_ft.py`: GPT-2をPEFTでtuningするサンプルコード。関数にしているだけ。data_collectorなどがないのでそのままでは使用不可。

## 環境構築
1. docker imageを使っています (dockerfileに書いてもOKです)。PyTorchのバージョンは2以降です。

```bash
docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
```

2. コンテナを作って入ります (省略)

3. (確か)gitとかが必要なのでインストールします (設定によってはコンテナにsudo権限が付与されているのでsudoは不要)
```bash
sudo apt update 
sudo apt upgrade -y
sudo apt install git
```

4. パッケージのインストール 
```bash
pip install -r requirements.txt
```

peftの実験はしていないので入っていません。入れる場合は
```bash
pip install peft==0.11.1
```

5. 現在のコードはアクセストークンをハードコーディングしていないので、環境変数に入れる

```bash
export HF_TOKEN="YOUR_ACCESS_TOKEN"
```

## 実行
`mmlu.py`を実行するだけ。`main`が実行されます。引数によって使用モデルが変わります。下の表を参照。

```bash
python mmlu.py 1
```

初回はモデルのダウンロードに時間がかかるのでバックグラウンド実行がおすすめ。
```bash
nohup python mmlu.py 1 > out.txt 2>&1 &
```
や
```bash
tmux new -s llm-test
python mmlu.py 1
```
など (tmuxは使ったことがないので違うかもしれないです)。

コマンド引数は以下の通り。数値で指定
| index | model name |
| ---- | ---- |
| 0 | microsoft/Phi-3-mini-128k-instruct |
| 1 | microsoft/Phi-3-small-128k-instruct |
| 2 | microsoft/Phi-3-medium-128k-instruct |
| 3 | Qwen/Qwen2-7B-Instruct |
| 4 | meta-llama/Meta-Llama-3-8B-Instruct |
| 5 | google/gemma-2-9b-it |
| 6 | google/gemma-2-27b-it |


## エラー
下のものに似たのが出た：もう1回実行すればできます。
> requests.exceptions.ChunkedEncodingError: ('Connection broken: IncompleteRead(749845505 bytes read, 4226853167 more expected)', IncompleteRead(749845505 bytes read, 4226853167 more expected))

## License 
各モデルのライセンスに従ってください。
