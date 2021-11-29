import json
import os
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja
from main import extract_phrase
import importlib
gpt2_score = importlib.import_module("gpt2-score")

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2ja-medium')
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--context', type=str, default='<|endoftext|>')
parser.add_argument('--num_generate', type=int, default=5)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--top_p', type=float, default=0)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_length', type=int, default=500)
parser.add_argument('--min_length', type=int, default=0)
args = parser.parse_args()

with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

if os.path.isfile(args.model+'/hparams.json'):
    with open(args.model+'/hparams.json', encoding='utf-8') as f:
        params = json.loads(f.read())
        hparams = HParams(**params)
elif 'small' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12
    })
elif 'medium' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
elif 'large' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1280,
      "n_head": 20,
      "n_layer": 36
    })
else:
    raise ValueError('invalid model name.')

length=hparams.n_ctx // 2
temperature=args.temperature
top_k=args.top_k
top_p=args.top_p
min_length=max(args.min_length,0)

def generate_one(sess, output, input):
    generated = ''
    pre_text = input if len(input)>0 else '<|endoftext|>'
    while True:
        context_tokens = enc.encode(pre_text)
        if len(context_tokens) > length:
            context_tokens = context_tokens[-length:]
        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })[:,len(context_tokens):]
        swd = enc.decode(out[0])
        cnt_kuten = 0
        kakko = True
        for i in range(len(swd)):
            if swd[i] == '「':
                kakko = False
            if swd[i] == '」':
                kakko = True
            if (swd[i] == '。' or swd[i] == '!' or swd[i] == ')' or swd[i] == '♬') and kakko:
                cnt_kuten += 1
            if cnt_kuten == 1:
                swd = swd[:i+1]
                break
        last = False
        if '<|endoftext|>' in swd:
            swd = swd.split('<|endoftext|>')[0]
            last = True
        last = True
            
        if len(swd) > 0:
            generated += swd

        if last or len(generated) > args.max_length:
            if len(generated) > 0:
                return generated[:args.max_length]
        else:
            pre_text = generated[-256:]

config = tf.ConfigProto()
if int(args.gpu) >= 0:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu

# 3つのセリフを代入
input_sentences = extract_phrase()
output_sentences = ''
output_list = []
CONJUNCTIONS = ['だから','そして','それで','さらに','このため','しかし','けれども','なぜなら','ところが']
for i in range(len(input_sentences)):
    with tf.Session(config=config,graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        output = sample_sequence(
            hparams=hparams, length=length,
            min_length=min_length, context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)

        if len(args.output_file) > 0:
            with open(args.output_file, 'w', encoding='utf-8') as of:
                for i in range(args.num_generate):
                    of.write(generate_one(sess, output, input_sentences[i])+'\n')
                    if i < args.num_generate-1:
                        of.write('========\n')
        else:
            output_sentences += input_sentences[i]
            output_generate = generate_one(sess, output, output_sentences)
            output_list.append(input_sentences[i]+output_generate)
            output_sentences += output_generate

sentence_candidates = []
for conjunction in CONJUNCTIONS:
    sentence = output_list[0] + conjunction + output_list[1]
    sentence_candidates.append(sentence)
req_sentence = ''
for sentence in sentence_candidates:
    req_sentence += sentence + '\n'
scores = gpt2_score.main(req_sentence)
max_index = scores.index(max(scores))
output_list[1] = CONJUNCTIONS[max_index] + output_list[1]

sentence_candidates = []
for conjunction in CONJUNCTIONS:
    sentence = output_list[1] + conjunction + output_list[2]
    sentence_candidates.append(sentence)
req_sentence = ''
for sentence in sentence_candidates:
    req_sentence += sentence + '\n'
scores = gpt2_score.main(req_sentence)
max_index = scores.index(max(scores))
output_list[2] = CONJUNCTIONS[max_index] + output_list[2]

output_result = ''
for i in range(len(output_list)):
    output_result += output_list[i]
print('output=', output_result)