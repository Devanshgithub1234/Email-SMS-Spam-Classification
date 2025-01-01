from flask import Flask,render_template,request
import requests

import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

application = Flask(__name__)
app=application
## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        sms=request.form.get('inp')
        sms=transform_text(sms)
        new_data_scaled=tfidf.transform([sms])
        result=model.predict(new_data_scaled)
        if result[0]==0:
            return render_template('home.html',result="Ham")
        else:
            return render_template('home.html',result="Spam")
        
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
