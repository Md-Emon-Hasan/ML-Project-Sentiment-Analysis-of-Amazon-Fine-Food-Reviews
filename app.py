from flask import Flask
from flask import render_template
from flask import request
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import pickle

app = Flask(__name__)

# Necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# load the pre-trained model and vectorizer
bow = pickle.load(open('./models/bow.pkl','rb'))
model = pickle.load(open('./models/model.pkl','rb'))

# stopwords
total_stopwords = set(stopwords.words('english'))

# subtract negative stop words like no, not, don't etc.. from total_stopwords
negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
final_stopwords = total_stopwords - negative_stop_words
final_stopwords.add("one")

# Remove unwanted words from reviews
# Ex. html tags, punctuation, stop words, etc..
stemmer = PorterStemmer()
HTMLTAGS = re.compile('<.*?>')
table = str.maketrans(dict.fromkeys(string.punctuation))
remove_digits = str.maketrans('','',string.digits)
MULTIPLE_WHITESPACE = re.compile(r'\s+')

def preprocessor(review):
    # remove html tags
    review = HTMLTAGS.sub(r'', review) 
    # remove puncutuation
    review = review.translate(table)
    # remove digits
    review = review.translate(remove_digits)
    # lower case all letters
    review = review.lower()
    # replace multiple white spaces with single space
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    # remove stop words
    review = [word for word in review.split()if word not in final_stopwords]
    # stemming
    review = ' '.join([stemmer.stem(word) for word in review])
    
    return review

@app.route('/',methods=['GET','POST'])
def index():
    result = None
    input_review = ''
    if request.method == 'POST':
        input_review = request.form['review']
        # 1. Preprocess the text
        transformed_sms = preprocessor(input_review)
        # 2. Vectorize the text
        vector_input = bow.transform([transformed_sms])
        # 3. Predict the result
        result = model.predict(vector_input)[0]
    
    return render_template('index.html',result=result,input_review=input_review)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)