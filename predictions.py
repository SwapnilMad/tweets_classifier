from sklearn.externals import joblib
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np

def get_lemma(sentence):
    word_list = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return output

loaded_log = joblib.load(open("linear.pkl", 'rb'))
loaded_naive = joblib.load(open("naive.pkl", 'rb'))

input = "I'm so lonely"

print(input)
perf = loaded_naive.predict_proba([get_lemma(input)])
y_pos = np.arange(len(loaded_naive.classes_))
plt.bar(y_pos, perf[0], align='center', alpha=0.5)
plt.xticks(y_pos, loaded_naive.classes_)
plt.ylabel('percentage')
plt.title('Percent Probability')


print('Naive Bayes', loaded_naive.predict([get_lemma(input)])[0])
print('SVC', loaded_log.predict([get_lemma(input)])[0])
plt.show()
