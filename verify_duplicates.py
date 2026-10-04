"""Audit the committed dataset; compare original and deduplicated LinearSVC runs."""
import json, hashlib, platform
from pathlib import Path
import pandas as pd
import sklearn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from utils import load_and_clean_data

df = load_and_clean_data('MeToo_tweets.csv')
analyzer = SentimentIntensityAnalyzer()
def label(text):
    score = analyzer.polarity_scores(text)['compound']
    return 'positive' if score >= .05 else 'negative' if score <= -.05 else 'neutral'
df['sentiment'] = df.clean_text.map(label)
result = {'dataset_sha256': hashlib.sha256(Path('MeToo_tweets.csv').read_bytes()).hexdigest(),
          'cleaned_rows':len(df), 'duplicate_entries':int(df.clean_text.duplicated().sum()),
          'unique_texts':int(df.clean_text.nunique()), 'python':platform.python_version(),
          'sklearn':sklearn.__version__, 'pandas':pd.__version__, 'vaderSentiment':'3.3.2', 'runs':[]}
for dedup in [False,True]:
    data = df.drop_duplicates('clean_text') if dedup else df
    for fit_before_split in [True,False]:
        vect = TfidfVectorizer(max_features=5000)
        x = vect.fit_transform(data.clean_text) if fit_before_split else data.clean_text
        xt,xv,yt,yv = train_test_split(x,data.sentiment,test_size=.2,random_state=42)
        if not fit_before_split:
            xt=vect.fit_transform(xt); xv=vect.transform(xv)
        model=LinearSVC(random_state=42)
        model.fit(xt,yt); pred=model.predict(xv)
        result['runs'].append({'deduplicated':dedup,'tfidf_fit_before_split':fit_before_split,
            'train_rows':len(yt),'test_rows':len(yv),'accuracy':accuracy_score(yv,pred),
            'macro_f1':f1_score(yv,pred,average='macro')})
Path('duplicate_audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
