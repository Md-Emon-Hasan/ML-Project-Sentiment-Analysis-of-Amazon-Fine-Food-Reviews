from unittest import result
from flask import Flask
from flask import render_template
from flask import request
import pickle
import nltk

app = Flask(__name__)

# Necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# load the pre-trained model and vectorizer
# bow = pickle.load(open('./models/bow.pkl','rb'))
# model = pickle.load(open('./models/model.pkl','rb'))

bow = pickle.load(open('C:/Users/emon1/OneDrive/Desktop/my projects/main projects/models/bow.pkl','rb'))
model = pickle.load(open('C:/Users/emon1/OneDrive/Desktop/my projects/main projects/models/model.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def index():
    result = None
    input_review = ''
    if request.method == 'POST':
        input_review = request.form['review']
        # 1. Vectorize the text
        vector_input = bow.transform([input_review])
        # 2. Predict the result
        result = model.predict(vector_input)[0]
    
    return render_template('index.html',result=result,input_review=input_review)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)