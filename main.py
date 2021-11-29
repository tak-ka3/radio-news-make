import os
import speech_recognition as sr
import glob
from pydub import AudioSegment
import summarization
from speech_recognition import UnknownValueError

def extract_phrase():
  radio_file = './laughter-detection/suda_komatsu.wav'
  if (glob.glob('./laughter-detection/tst_wave/*')):
    os.system('rm ./laughter-detection/tst_wave/laugh_*.wav')
  # 笑いがある部分の音声ファイルを出力
  os.system('srun -p p -t 10:00 --gres=gpu:1 --pty python3 ./laughter-detection/segment_laughter.py --input_audio_file={} --output_dir=./laughter-detection/tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.2'.format(radio_file))
  files = glob.glob('./laughter-detection/tst_wave/*')
  r = sr.Recognizer()
  laugh_list = []
  # 笑いがある部分を文字起こし
  for f in files:
    with sr.AudioFile(f) as source:
      try: 
        audio = r.record(source)
        text = r.recognize_google(audio, language='ja-JP')
        new_text = text.replace(' ', '。') + '。'
        laugh_list.append(new_text)
      except UnknownValueError:
        continue
  laugh_str = '。'.join(laugh_list)
  new_laugh_list = laugh_str.split('。')

  sound = AudioSegment.from_file(radio_file, format='wav')
  radio_length = len(sound)
  # 新しくディレクトリを作る
  sound_split_dir = './laughter-detection/outsound_split'
  os.system('mkdir -p {}'.format(sound_split_dir))
  # 一回の分割のms
  # 60000以上だとif not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()
  # というようなエラーが出てしまう
  split_ms = 50000
  radio_length_split = radio_length//split_ms
  # すでに分割wavファイルがあったら削除する
  if (glob.glob(sound_split_dir+'/*')):
    os.system('rm {}/sound_split_*.wav'.format(sound_split_dir))
  # 音声ファイルを分割して別名で保存する。
  for i in range(radio_length_split):
    sound_split = sound[i*split_ms:(i+1)*split_ms]
    sound_split.export('{}/sound_split_{}.wav'.format(sound_split_dir, i), format='wav')
  # 最後の分割ファイルの作成
  sound_split = sound[radio_length_split*split_ms:radio_length]
  sound_split.export('{}/sound_split_{}.wav'.format(sound_split_dir, radio_length_split), format='wav')
  sentences_list = []
  # # 文字起こしして出力
  for i in range(radio_length_split+1):
    with sr.AudioFile(sound_split_dir+'/sound_split_{}.wav'.format(i)) as source:
      try:
        audio = r.record(source)
        text = r.recognize_google(audio, language='ja-JP')
        new_text = text.replace(' ', '。') # 空白を句点に置換する
        sentences_list.append(new_text)
      except UnknownValueError:
        continue

  sentences_str = ''.join(sentences_list) + '。'
  sentences_new_list = sentences_str.split('。')

  # 文字起こしから大切な文だけを出力
  result_dict = summarization.summarization(sentences_str)
  result_length = len(result_dict['summarize_result'])
  # keyがフレーズで、valueがスコアの辞書を作る
  new_score_dict = {}
  for i in range(result_length):
    new_score_dict[result_dict['summarize_result'][i]] = [result_dict['scoring_data'][i][1], i]
  
  new_final_laugh_list = []
  for word in new_laugh_list:
    if len(word) > 2:
      new_final_laugh_list.append(word)
  # print(new_score_dict)
  for k, v in new_score_dict.items():
    for word in new_final_laugh_list:
      if word in k:
        new_score_dict[k][0] *= 1.5
        break
  # scoreに応じてsortする
  sorted_phrase = sorted(new_score_dict.items(), key=lambda x:x[1][0], reverse=True)

  new_sorted_phrase = []
  for i in range(3):
    new_sorted_phrase.append(sorted_phrase[i])
  time_sorted_phrase = sorted(new_sorted_phrase, key=lambda x: x[1][1])
  # 取り出すフレーズの数
  phrase_num =3
  phrase_list = []
  
  # sortに従って上位いくつかを出力する
  for i in range(phrase_num):
    if time_sorted_phrase[i][0][-1] == '\n':
      phrase_list.append(time_sorted_phrase[i][0][:-1])
    else:
      phrase_list.append(time_sorted_phrase[i][0])
  # フレーズを整形する
  phrase_list[0] = '菅田将暉は、「{}」と'.format(phrase_list[0])
  phrase_list[1] = '「{}」と'.format(phrase_list[1])
  phrase_list[2] = '「{}」と'.format(phrase_list[2])
  return phrase_list