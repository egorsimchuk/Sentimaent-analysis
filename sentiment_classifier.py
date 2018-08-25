from __future__ import print_function

from sklearn.externals import joblib
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./models/clf.pkl")
        self.vectorizer = joblib.load("./models/vectorizer.pkl")
        self.classes_dict = {0: "отрицательный", 1: "положительный", -1: "ошибка"}
        self.numbers_str = '0123456789'
        self.punc_translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        self.num_translator = str.maketrans(self.numbers_str, ' ' * len(self.numbers_str))
        self.short_word_len = 1
        self.stemmer = RussianStemmer()
        self.stop_words = stopwords.words('russian') + ['br']




    def predprocess_text(self, text):
        text = text.lower().translate(self.punc_translator).translate(self.num_translator)
        text = ' '.join(self.stemmer.stem(word) for word in text.split())
        return text.strip()

    def predict_text(self, text):
        text = self.predprocess_text(text)
        try:
            print(text)
            vectorized = self.vectorizer.transform([self.predprocess_text(text)])
            return self.model.predict(vectorized)[0]
        except:
            print("prediction error")
            return -1


    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        return self.classes_dict[prediction]