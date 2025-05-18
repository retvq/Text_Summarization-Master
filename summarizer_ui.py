import streamlit as st
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import math

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(filtered_words)

class sentence:
    def __init__(self):
        self.sentence = None
        self.tokens = []
        self.weights = {}

class paragraph:
    def __init__(self):
        self.paragraph = None
        self.tokens = []
        self.weights = {}
        self.quota = 0
        self.sentences = []
        self.sentences_keep = []

class fractal:
    def __init__(self, text):
        self.paragraphs = []
        self.weights = FreqDist(nltk.word_tokenize(preprocess(text)))
        self.sentence_keep = []
        self.sentences_sorted = []

class fractalSummary:
    def __init__(self, text, quota):
        self.text = text
        self.quota = quota
        self.summary = []
        self.fractal = fractal(text)
        self.paragraphs = text.split('\n\n')
        self.s_weight = 0
        self.p_weight = 0
        self.pindex = 0
        self.ptotal = len(self.paragraphs)
        self.sindex = 0
        self.stotal = 0
        self.np_weight = 0
        self.ns_weight = 0
        self.sentences_keep = []

    def calculate_relative_frequence(self, sentence_token, weights):
        return {w: freq for w, freq in weights.items() if w in sentence_token}

    def fractal_representation(self):
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        for each_paragraph in self.paragraphs:
            buffer_p = paragraph()
            buffer_p.paragraph = each_paragraph
            buffer_p.tokens = nltk.word_tokenize(preprocess(each_paragraph))
            buffer_p.weights['words'] = FreqDist(buffer_p.tokens)
            buffer_p.weights['total'] = {'words': 0, 'sentences': 0}
            sentence_splitter = PunktSentenceTokenizer(punkt_param)
            sentences = sentence_splitter.tokenize(each_paragraph)
            for each_sentence in sentences:
                self.stotal += 1
                buffer_s = sentence()
                buffer_s.sentence = each_sentence
                buffer_s.tokens = nltk.word_tokenize(preprocess(each_sentence))
                if buffer_s.tokens:
                    buffer_s.weights['sentence'] = FreqDist(buffer_s.tokens)
                    buffer_s.weights['paragraph'] = self.calculate_relative_frequence(buffer_s.tokens, buffer_p.weights['words'])
                    buffer_s.weights['document'] = self.calculate_relative_frequence(buffer_s.tokens, self.fractal.weights)
                    buffer_s.weights['total'] = {
                        'sentence': 1,
                        'paragraph': sum(buffer_s.weights['paragraph'].values()),
                        'document': sum(buffer_s.weights['document'].values())
                    }
                    self.s_weight += buffer_s.weights['total']['document']
                    buffer_p.weights['total']['sentences'] += buffer_s.weights['total']['document']
                    buffer_p.sentences.append(buffer_s)
            buffer_p.weights['total']['words'] = sum(buffer_p.weights['words'].values())
            self.fractal.paragraphs.append(buffer_p)
            self.pindex += 1

    def normalize(self):
        for each_paragraph in self.fractal.paragraphs:
            each_paragraph.weights['total']['normalized'] = each_paragraph.weights['total']['sentences'] / float(self.s_weight)
            self.np_weight += each_paragraph.weights['total']['normalized']
            each_paragraph.quota = round(self.quota * each_paragraph.weights['total']['normalized'])

            sentences_sorted = []
            for idx, each_sentence in enumerate(each_paragraph.sentences):
                each_sentence.weights['total']['sentence_normalized'] = each_sentence.weights['total']['document'] / float(self.s_weight)
                self.ns_weight += each_sentence.weights['total']['sentence_normalized']
                sentences_sorted.append({
                    'weight': each_sentence.weights['total']['sentence_normalized'],
                    'text': each_sentence.sentence,
                    'index': idx
                })
            sentences_sorted.sort(key=lambda x: x['weight'], reverse=True)
            sentences_sorted = sentences_sorted[:int(each_paragraph.quota)]
            sentences_sorted.sort(key=lambda x: x['index'])
            for s in sentences_sorted:
                self.sentences_keep.append(s['text'])

        self.summary = self.sentences_keep[:self.quota]

class WordFrequency:
    def __init__(self, text, quota):
        self.text = text
        self.summary = []
        self.quota = quota

    def calculate_relative_frequence(self, tokens, weights):
        return {w: weights[w] for w in tokens if w in weights}

    def summarize(self):
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentences = sentence_splitter.tokenize(self.text)
        weights = FreqDist(nltk.word_tokenize(preprocess(self.text)))

        sentence_data = []
        for idx, s in enumerate(sentences):
            tokens = nltk.word_tokenize(preprocess(s))
            freq = self.calculate_relative_frequence(tokens, weights)
            sentence_data.append({'text': s, 'index': idx, 'score': sum(freq.values())})
        sentence_data.sort(key=lambda x: x['score'], reverse=True)
        top = sorted(sentence_data[:self.quota], key=lambda x: x['index'])
        self.summary = [s['text'] for s in top]

class sinFrequencySummary(WordFrequency):
    def summarize(self):
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentences = sentence_splitter.tokenize(self.text)
        weights = FreqDist(nltk.word_tokenize(preprocess(self.text)))
        sentence_data = []
        total = len(sentences)
        for idx, s in enumerate(sentences):
            tokens = nltk.word_tokenize(preprocess(s))
            freq = self.calculate_relative_frequence(tokens, weights)
            score = sum(freq.values())
            sin_weight = (1 - math.sin(idx * (math.pi / total))) / 2 + 1
            sentence_data.append({'text': s, 'index': idx, 'score': score * sin_weight})
        sentence_data.sort(key=lambda x: x['score'], reverse=True)
        top = sorted(sentence_data[:self.quota], key=lambda x: x['index'])
        self.summary = [s['text'] for s in top]

class sinWordFrequencySummary(WordFrequency):
    def summarize(self):
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentences = sentence_splitter.tokenize(self.text)
        base_weights = FreqDist(nltk.word_tokenize(preprocess(self.text)))
        total = len(sentences)

        for idx, s in enumerate(sentences):
            sin_weight = (1 - math.sin(idx * (math.pi / total))) + 1
            tokens = nltk.word_tokenize(preprocess(s))
            for t in tokens:
                if t in base_weights:
                    base_weights[t] *= sin_weight

        sentence_data = []
        for idx, s in enumerate(sentences):
            tokens = nltk.word_tokenize(preprocess(s))
            freq = self.calculate_relative_frequence(tokens, base_weights)
            sentence_data.append({'text': s, 'index': idx, 'score': sum(freq.values())})

        sentence_data.sort(key=lambda x: x['score'], reverse=True)
        top = sorted(sentence_data[:self.quota], key=lambda x: x['index'])
        self.summary = [s['text'] for s in top]


# --------------------------- Streamlit UI ----------------------------

st.title("🧠 Text Summarizer")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

quota = st.slider("Select number of sentences for the summary", min_value=1, max_value=20, value=6)

method = st.selectbox("Choose summarization method", [
    "Fractal Summary", "Word Frequency Summary", "Sin Frequency Summary", "Sin Word Frequency Summary"
])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    if st.button("Generate Summary"):
        if method == "Fractal Summary":
            fs = fractalSummary(text, quota)
            fs.fractal_representation()
            fs.normalize()
            summary = fs.summary

        elif method == "Word Frequency Summary":
            wf = WordFrequency(text, quota)
            wf.summarize()
            summary = wf.summary

        elif method == "Sin Frequency Summary":
            sf = sinFrequencySummary(text, quota)
            sf.summarize()
            summary = sf.summary

        elif method == "Sin Word Frequency Summary":
            swf = sinWordFrequencySummary(text, quota)
            swf.summarize()
            summary = swf.summary

        st.subheader("📄 Summary:")
        for i, line in enumerate(summary, 1):
            st.markdown(f"**{i}.** {line}")
