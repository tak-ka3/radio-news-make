from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

def summarization(document):
  auto_abstractor = AutoAbstractor()
  auto_abstractor.tokenizable_doc = MeCabTokenizer()
  auto_abstractor.delimiter_list = ['。', '\n']
  abstractable_doc = TopNRankAbstractor()
  result_dict = auto_abstractor.summarize(document, abstractable_doc)

  return result_dict