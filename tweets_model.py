import os,glob,csv
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.pipeline import Pipeline
import emoji
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.externals import joblib

folder_path = '####################################'

def get_lemma(sentence):
    word_list = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return output

def clean_text(word_list):
    word_list = emoji.demojize(word_list)
    word_list = word_list.lower()

    word_list = word_list.replace(r"::", ": :")
    word_list = word_list.replace(r"[^a-z\':_]", " ")
    word_list = word_list.replace(r"(can't|cannot)", 'can not ')
    word_list = word_list.replace(r"n't", ' not')
    word_list = re.sub(r"i'm", "i am ", word_list)
    word_list = re.sub(r"\'re", " are ", word_list)
    word_list = re.sub(r"\'d", " would ", word_list)
    word_list = re.sub(r"\'ll", " will ", word_list)
    word_list = re.sub(r"\'scuse", " excuse ", word_list)
    word_list = re.sub('\W', ' ', word_list)
    word_list = re.sub('\s+', ' ', word_list)
    word_list = re.sub('_', ' ', word_list)
    word_list = word_list.strip(' ')
    return get_lemma(word_list)

category=[]
tweets=[]
stopwords=nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')

for filename in glob.glob(os.path.join(folder_path, '*.csv')):
    try:
        with open(filename, 'r', encoding='utf8') as f:
            csv_file = csv.DictReader(f)
            for row in csv_file:
               if row and row['truncated'].lower()=='false':
                    tweets.append(clean_text(row['text']))
                    emo=filename.split('_')[1]
                    if(emo == 'happy' or emo == 'glad' or emo == 'love' or emo == 'excited' or emo == 'thankful' or emo == 'lit' or emo == 'grateful'):
                        category.append('happy')
                    elif(emo == 'sad' or emo == 'depressed' or emo == 'disappointed' or emo == 'upset'):
                        category.append('sad')
                    elif(emo == 'angry' or emo == 'rage' or emo == 'hate' or emo == 'upset'):
                        category.append('angry')
                    elif(emo == 'scared' or emo == 'afraid' or emo == 'disgusting' or emo == 'gross' or emo == 'surprised'):
                        category.append('scared')
                    else:
                        category.append('none')
    except (OSError ,RuntimeError, TypeError, NameError):
          pass

print('size',len(tweets))
data = {'tweets':tweets,
        'category':category}

feature = pd.DataFrame({'tweets': tweets})
targets = pd.DataFrame({'category': category})

training_examples, validation_examples, training_targets, validation_targets = train_test_split(feature, targets, test_size=0.20)

X_train, X_test, y_train, y_test = train_test_split(tweets, category, test_size=0.20)

NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stopwords)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])

SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stopwords)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

NB_pipeline.fit(X_train, y_train)
joblib.dump(NB_pipeline, 'naive.pkl', compress = 1)
prediction = NB_pipeline.predict(X_test)
print('Test accuracy Naive is {}'.format(accuracy_score(y_test, prediction)))

SVC_pipeline.fit(X_train, y_train)
joblib.dump(SVC_pipeline, 'linear.pkl', compress = 1)
predictionlinear = SVC_pipeline.predict(X_test)
print('Test accuracy SVC is {}'.format(accuracy_score(y_test, predictionlinear)))