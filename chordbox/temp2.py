import os
import json
import numpy as np
import librosa
from pytube import YouTube
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
import requests
from celery import Celery
import vamp
import input_transform as it
import uuid
import ffmpeg

def transform_url(url):    
    #유튜브 전용 인스턴스 생성
    yt = YouTube(url)
    yt.streams.filter(only_audio=True).all()
    # 특정영상 다운로드
    name = uuid.uuid1()
    yt.streams.filter(only_audio=True).first().download()
    os.rename(yt.streams.first().default_filename[:-4] + 'mp4', str(name) + '.mp4')    
    return yt.streams.first().default_filename[:-4] + 'mp4', str(name) + '.mp4'

app = Flask(__name__)
BROKER_URL = 'redis://localhost:6379'
CELERY_RESULTS_BACKEND = 'redis://localhost:6379'
celery = Celery('run_test2', broker=BROKER_URL, backend=CELERY_RESULTS_BACKEND)

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
            print("File upload failed")
            return {'status': '400'}

        mp4_path = path
        n, s, sh, c = it.get_chromagram(mp4_path)
        result_times, result_chords = it.predict(n, s, sh, c)
        os.remove(mp4_path)

    return {'_id': _id, 'times': result_times, 'chords': result_chords}


@app.route('/predict_url', methods=['POST'])
def predict_url():
    if request.method == 'POST':
        form = request.form
        try:
            _id = form['_id']
            url = form['url']
            print('URL received')
        except:
            print('URL unavailable')
            return {'status': '400'}
        chordfilename, unique_id = transform_url(url)
        mp4_path = '/home/chordbox2021/chordbox/' + unique_id
        get_response.delay(_id, mp4_path, chordfilename)

    return {'status': '200'}


@celery.task(serializer='json')
def get_response(_id, mp4_path, chordfilename):
    n, s, sh, c = it.get_chromagram(mp4_path)
    result_times, result_chords = it.predict(n, s, sh, c)

    mp3_path = mp4_path[:-3] + "mp3"
    ffmpeg.input(mp4_path).output(mp3_path).run(overwrite_output=False)

    mp3 = open(mp3_path, 'rb')
    mp3_file_name = mp3_path.split('/')[-1]
    print(mp3_file_name)

    multipart_form_data = {
            'audiofile' : (mp3_file_name, mp3, 'audio/mpeg'),
            'chordfilename' : (None, chordfilename), 
            '_id': (None, _id),
            'chords': (None, result_chords),
            'times': (None, result_times)
            }

    try:
        res = requests.put('http://35.193.88.18:3333/resultUrl', files=multipart_form_data)
        print("mp4 upload requested.")
        os.remove(mp4_path)
    except requests.exceptions.HTTPError as errb:
        print("Http Error : ", errb)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting : ", errc)



if __name__=='__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)
