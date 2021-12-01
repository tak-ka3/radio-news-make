# 芸能ニュースメーカー(eeic 人工知能演習)

## Abstract
ラジオを元にした芸能ニュースの記事の自動生成

## Preparation
pythonのバージョンをminicondaなどを利用して、`3.7.10`に設定してください。（おそらく3.7以上だといける）

## How to run
```bash
$ cd radio-news-make
$ pip3 install -r requirements.txt
$ pip3 install -r laughter-detection/requirements.txt
$ wget "https://drive.google.com/uc?export=download&id=1hryE15ky-uAFy9pFPXix5d5epJ23HUzR" -O ./laughter-detection/suda_komatsu.wav
$ wget https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2
$ tar xvfj gpt2ja-medium.tar.bz2
$ python3 ./Japanese-BPEEncoder/encode_bpe.py --src_dir entertainment --dst_file finetune
$ python3 run_finetune.py --base_model gpt2ja-medium --dataset finetune.npz --run_name gpt2ja-finetune_run1
$ ./1gpu-test.sh
```
