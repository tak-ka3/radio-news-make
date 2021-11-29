# 芸能ニュースメーカー(eeic 人工知能演習)

## 概要
ラジオを元にした芸能ニュースの記事の自動生成

## How to run
```bash
$ cd gpt2-japanese/
$ git clone git@github.com:jrgillick/laughter-detection.git
$ mv laugh_segmenter.py ./laughter-detection/laugh_segmenter.py
$ mv segment_laughter.py ./laughter-detection/segment_laughter.py
$ wget "https://drive.google.com/uc?export=download&id=1hryE15ky-uAFy9pFPXix5d5epJ23HUzR" -O ./laughter-detection/suda_komatsu.wav
$ wget https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2
$ tar xvfj gpt2ja-medium.tar.bz2
$ git clone https://github.com/tanreinama/Japanese-BPEEncoder.git
$ python ./Japanese-BPEEncoder/encode_bpe.py --src_dir entertainment --dst_file finetune
$ python3 run_finetune.py --base_model gpt2ja-medium --dataset finetune.npz --run_name gpt2ja-finetune_run1
$ ./1gpu-test.sh
```
