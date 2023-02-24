import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

class SimpleTokenizer:
    """Simple tokenization including stemming and removing stop words. Used for logistic regression."""

    def __init__( self, remove_stopwords = True, apply_stemming = True,
    ):
        self.remove_stopwords = remove_stopwords 
        self.apply_stemming = apply_stemming


    def tokenize(self, df_col: pd.Series) -> pd.Series:
        """
        Takes as input a DataFrame column as string. 
        Return tokenized values.    
        """

        tokenizer = RegexpTokenizer("[a-zA-Z@]+") # We only want words in text as punctuation and numbers are not helpful
        df_col = df_col.apply(tokenizer.tokenize)

        
        if self.remove_stopwords:
            stop = stopwords.words('english') 
            df_col = df_col.apply(lambda x: ' '.join([word for word in x if word not in stop]))

        if self.apply_stemming:
            ss = SnowballStemmer("english")
            df_col = df_col.apply(lambda x: ' '.join([ss.stem(word) for word in x.split()]))
        
        return df_col

    def __call__(self, df_col: pd.Series) -> pd.Series:
        return self.tokenize(df_col)