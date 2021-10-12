
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import pickle
import flask
from TwitterBeautifulSoupSelenium import veri_cek

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/')#Arayüz
def main():
    return render_template('main.html')

#Kullanıcıdan alınan kullanıcı adı twitter adresi ile birleşiğ veri_cek modülüne gidiyor
@app.route('/predict',methods=['GET','POST'])
def predict():
    user =request.get_data(as_text=True)[5:]
    url ='https://twitter.com/'+user
    #url = "https://twitter.com/BBCNews"
    news = veri_cek(url)#gelen veriler news değişkenine atılıyor dType: List
    #print(news)
    #print("Veri tipi: ",type(news))
    #Tahmin fonksiyonuna giriyor
    pred = model.predict(news)
    print("''''''''''''''''''''''''''''",pred)
    return render_template('main.html', prediction_text='The news is "{}"'.format(news)+'"{}"'.format(i))

if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
