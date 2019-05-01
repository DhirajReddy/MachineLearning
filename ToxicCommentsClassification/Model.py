import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
import pickle

class GeneralModel:

    def __init__(self):
        self.vectorizers = []
        self.model = None


    def load_from_pickle(self, file):
        return pickle.load(open(file, 'rb'))


    def load_model(self, pickleFile):
        self.model = self.load_from_pickle(pickleFile)


    def load_vectorizers(self, pickleFiles):
        for pf in pickleFiles:
            self.vectorizers.append(self.load_from_pickle(pf))


    def predict(self, documents):
        """
        documents : array of documents
        """
        features = []
        for vectorizer in self.vectorizers:
            features.append(vectorizer.transform(documents))
        
        features = hstack(features)

        return self.model.predict(features)
