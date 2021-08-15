from flask import Flask,render_template,url_for,request
import pickle

# load the model from disk
clf=pickle.load(open('multi-label-classification.pkl', 'rb'))
tfidf = pickle.load(open('td-idf.pkl', 'rb'))
multilabel=pickle.load(open('multi-label.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message =request.form['message']
        data = [message]
        xt = tfidf.transform(data)
        output = clf.predict(xt)
        output=multilabel.inverse_transform(output)
        #output = tfidf.inverse_transform(clf.predict(xt))
        print(output[0])
        return render_template('result.html',prediction = output[0])

if __name__ == '__main__':
	app.run(debug=True)
