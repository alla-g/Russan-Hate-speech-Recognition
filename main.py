import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.metrics import classification_report, plot_confusion_matrix, \
confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.dataset import read_dataset, balance_dataset
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.sentiment_analyzer import SentimentAnalyzer

nltk.download('punkt')
nltk.download('stopwords')

def plot_matrix(golds, preds):
    cm = confusion_matrix(golds, preds)
    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('True Values ')

    ax.xaxis.set_ticklabels(['Not toxic','Toxic'])
    ax.yaxis.set_ticklabels(['Not toxic','Toxic'])
    
    print(f'МАТРИЦА: {cm}')
    plt.show()

if __name__ == "__main__":

    # read the dataset
    df = read_dataset()

    # make the dataset balanced (stratify)
    df = balance_dataset(df)

    # preprocessing stage 1: Regular expressions & Filters
    print("preprocessing stage 1..")
    preprocessor = Preprocessor()
    df['text'] = df['text'].apply(preprocessor.preprocess)

    # extract the important features
    print("extraction of features..")
    feature_extractor = FeatureExtractor()
    df['offensive_words_count'] = df['text'].apply(feature_extractor.find_offensive_words)
    df['caps_words_count'] = df['text'].apply(feature_extractor.find_capsed_words)

    # if there is no sentiment data, label it automatically
    if not df['sentiment'].any():
        print("initiating sentiment analyzer..")
        df = SentimentAnalyzer().sentiment_label_dataframe(df)

    # acquire TD-IDF features
    print("acquring TF-IDF features..")
    tfidf, vocab, idf_dict, vectorizer = preprocessor.get_TFIDF_features(df, filter_stopwords=False)

    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    df['text'] = df['text'].apply(preprocessor.tokenize)
    df['text'] = df['text'].apply(preprocessor.lemmatize)

    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(df)
    M = np.concatenate([tfidf, feats], axis=1)

    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    # MODEL #

    X = DataFrame(M)
    y = df['hate_speech'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    print("training model..")
    model = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr',
                      max_iter=3000, dual=False)
    model.fit(x_train, y_train)

    ### ADD MY TEST CORPORA ###
    uncorrected = pd.read_csv('/content/toxicity-detection-thesis/data/uncorrected_data.tsv', sep='\t')
    corrected = pd.read_csv('/content/toxicity-detection-thesis/data/corrected_data.tsv', sep='\t')
    
    uncorrected['text'] = uncorrected['text'].apply(preprocessor.preprocess)
    corrected['text'] = corrected['text'].apply(preprocessor.preprocess)

    # extract the important features
    print("extraction of features..")
    feature_extractor = FeatureExtractor()
    uncorrected['offensive_words_count'] = uncorrected['text'].apply(feature_extractor.find_offensive_words)
    uncorrected['caps_words_count'] = uncorrected['text'].apply(feature_extractor.find_capsed_words)

    corrected['offensive_words_count'] = corrected['text'].apply(feature_extractor.find_offensive_words)
    corrected['caps_words_count'] = corrected['text'].apply(feature_extractor.find_capsed_words)

    # if there is no sentiment data, label it automatically
    print("initiating sentiment analyzer..")
    uncorrected = SentimentAnalyzer().sentiment_label_dataframe(uncorrected)
    corrected = SentimentAnalyzer().sentiment_label_dataframe(corrected)

    # acquire TD-IDF features for uncor
    print("acquring TF-IDF features for uncor..")
    tfidf, vocab, idf_dict, _ = preprocessor.get_TFIDF_features(uncorrected,
                                                                vectorizer=vectorizer,
                                                                filter_stopwords=False)
    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    uncorrected['text'] = uncorrected['text'].apply(preprocessor.tokenize)
    uncorrected['text'] = uncorrected['text'].apply(preprocessor.lemmatize)
    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(uncorrected)
    M = np.concatenate([tfidf, feats], axis=1)
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    X_uncor = DataFrame(M)

    # acquire TD-IDF features for cor
    print("acquring TF-IDF features for uncor..")
    tfidf, vocab, idf_dict, _ = preprocessor.get_TFIDF_features(corrected,
                                                                vectorizer=vectorizer,
                                                                filter_stopwords=False)
    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    corrected['text'] = corrected['text'].apply(preprocessor.tokenize)
    corrected['text'] = corrected['text'].apply(preprocessor.lemmatize)
    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(corrected)
    M = np.concatenate([tfidf, feats], axis=1)
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    X_cor = DataFrame(M)

    # конец препроца #

    y_uncor = uncorrected['toxicity'].astype(int)
    y_cor = corrected['toxicity'].astype(int)

    ### PREDICT ON MY TEST ###
    print('predicting on uncorrected')
    y_preds = model.predict(X_uncor)
    report = classification_report(y_uncor, y_preds)
    print(report)
    plot_confusion_matrix(model, X_uncor, y_uncor)
    plot_matrix(y_uncor, y_preds)

    print('predicting on corrected')
    y_preds = model.predict(X_cor)
    report = classification_report(y_cor, y_preds)
    print(report)
    plot_confusion_matrix(model, X_cor, y_cor)
    plot_matrix(y_cor, y_preds)
