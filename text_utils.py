import sys
import re
import unicodedata as ud

stopwords_path = 'vietnamese-stopwords.txt'
stopwords = set()

with open(stopwords_path, encoding='utf-8') as file:
    for line in file:
        li = line.strip()
        if not li.startswith('#'):
            word = '_'.join(li.split())
            stopwords.add(word)

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stopwords])

def remove_punc(text):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if ud.category(chr(i)).startswith('P'))
    return text.translate(tbl)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_numbers(text):
    pattern = r'[0-9]+' 
    return re.sub(pattern, '', text)
