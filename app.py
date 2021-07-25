from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

clf = pickle.load(open('models/clf.sav','rb'))
cv = pickle.load(open('models/cv.sav','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    X = cv.transform([email]).toarray()
    y_pred = clf.predict(X)
    if y_pred[0] == '0':
        response = -1
    else:
        response = 1
    return render_template('index.html',response=response)

if __name__ == "__main__":
    app.run(debug=True)
