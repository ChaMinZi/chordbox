import os
import json
import numpy
import librosa
from pytube import YouTube
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
import requests
from celery import Celery

def transform_url(name, url):
    filename = name + '.mp4'    
    #유튜브 전용 인스턴스 생성
    par = "https://www.youtube.com/watch?v=X-yIEMduRXk"
    yt = YouTube(par)
    yt.streams.filter(only_audio=True).all()
    # 특정영상 다운로드
    yt.streams.filter(only_audio=True).first().download()
    os.rename(yt.streams.first().default_filename[:-4]+'mp4', filename)    
    return filename

def get_chromagram(filename):
    sr = 16000
    y, _ = librosa.load(filename, duration=7, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

app = Flask(__name__)
BROKER_URL = 'redis://localhost:6379'
CELERY_RESULTS_BACKEND = 'redis://localhost:6379'
celery = Celery('run_test', broker=BROKER_URL, backend=CELERY_RESULTS_BACKEND)
        #celery.config_from_object('celeryconfig')
os.system('celery -A run_test.celery worker --loglevel=info')

@app.route('/')
def hello():
 return 'Welcome to ChordBox'

@app.route('/predict_wav', methods = ['POST'])
def predict_wav():
    if request.method == 'POST':
        form = request.form
        files = request.files

        try:
            _id = form['_id']
            wav = files['file']
        
            wav_name = secure_filename(wav.filename)
            path = '/home/chordbox2021/chordbox/file/' + wav_name
            wav.save(path)
            print(f'File upload completed {path}')
        except:
            print("Failed file upload")
        
        mp4_path = '/home/chordbox2021/chordbox/music/adele_easy_on_me.mp4'
        get_response.delay( _id, mp4_path)

        return {'status': '200'}
        """
        mp4 = open('/home/chordbox2021/chordbox/music/adele_easy_on_me.mp4', 'rb')
        files = {'audiofile': ('adele_easy_on_me', mp4)}
        payload = { '_id': _id, 'chords': 'test', 'times': 'test'}
        req = { '_id': _id, 'chords': 'test', 'times': 'test', 'audiofile': ('adele_easy_on_me', mp4)}

        multipart_form_data = {
                'audiofile' : ('musicresult.mp4', open('/home/chordbox2021/chordbox/music/test_final.mp4', 'rb'), 'audio/mp4'),
            '_id': (None, _id),
            'chords': (None, 'test'),
            'times': (None, 'test')
        }

        try:
            res = requests.put('http://35.193.88.18:3333/resultUrl', files=multipart_form_data)
        except requests.exceptions.HTTPError as errb:
            print("Http Error : ", errb)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting : ", errc)
        """
    
    #return make_response(json.dumps(res, ensure_ascii=False))

@app.route('/predict_url2', methods=['POST'])
def predict_url2():
    if request.method == 'POST':
        form = request.form
        try:
            _id = form['_id']
            url = form['url']
        except:
            print('URL unavailable')
        
        mp4 = open('/home/chordbox2021/chordbox/music/adele_easy_on_me.mp4', 'rb')
        get_response.delay(_id, mp4)
        return {'status': '200'}
        
@celery.task(serializer='json')
def get_response(_id, mp4_path):
    mp4 = open(mp4_path, 'rb')
    files = {'audiofile': ('adele_easy_on_me', mp4)}
    payload = { '_id': _id, 'chords': 'test', 'times': 'test'}
    req = { '_id': _id, 'chords': 'test', 'times': 'test', 'audiofile': ('adele_easy_on_me', mp4)}

    multipart_form_data = {
            'audiofile' : ('musicresult.mp4', open('/home/chordbox2021/chordbox/music/test_final.mp4', 'rb'), 'audio/mp4'),
            '_id': (None, _id),
            'chords': (None, 'test'),
            'times': (None, 'test')
            }

    try:
        res = requests.put('http://35.193.88.18:3333/resultUrl', files=multipart_form_data)
    except requests.exceptions.HTTPError as errb:
        print("Http Error : ", errb)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting : ", errc)


@app.route('/predict_url', methods=['POST'])
def predict_url():
    if request.method == 'POST':
        data = request.get_json()
        res = { 'id' : 'test', 'chord' : data['url'] } 
        res = make_response(json.dumps(res, ensure_ascii=False))
        res.headers['Content-Type'] = 'application/json'
        return res

if __name__=='__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)
