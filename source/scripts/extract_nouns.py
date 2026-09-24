import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag
from nltk.corpus import comtrans

text = "Questa è una frase di esempio che mostra il processo di tokenizzazione."

tokens = word_tokenize(text, language='italian')
tagged_tokens = pos_tag(tokens)

print(tagged_tokens)