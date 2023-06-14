from flask import Flask, Blueprint, render_template, request, flash, redirect, url_for, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from os import path
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from sqlalchemy.sql import func
from google.cloud import storage
#Importing library
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
from google.cloud import translate_v2 as translate

import contractions
import spacy
import re
import string
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#recommendation
import pandas as pd
import numpy as np

db = SQLAlchemy()
DB_NAME = "database.db"



def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    #from .views import views
    #from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    #from .models import User, Diary
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('web/' + DB_NAME):
        db.create_all(app=app)
        print('Database Created!')

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return {'status': 'success', 'message': 'Login berhasil'}
                #return redirect(url_for('views.home'))

            else:
                flash('Incorrect password, try again.', category='error')
                return {'status': 'error', 'message': 'Incorrect password'}
        else:
            flash('Email does not exist.', category='error')
            return {'status': 'error', 'message': 'Email does not exist.'}
    
    #return render_template("login.html", user=current_user)


@auth.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return {'status': 'success', 'message': 'Logout berhasil'}


@auth.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
            return {'status': 'error', 'message': 'Email already exists.'}
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
            return {'status': 'error', 'message': 'Email must be greater than 3 characters.'}
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
            return {'status': 'error', 'message': 'First name must be greater than 1 character.'}
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
            return {'status': 'error', 'message': 'Passwords don\'t match.'}
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
            return {'status': 'error', 'message': 'Password must be at least 7 characters.'}
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return {'status': 'success', 'message': 'Registrasi Berhasil'}

    #return render_template("sign_up.html", user=current_user)

class Diary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    Anxiety = db.Column(db.Float)
    Depresi = db.Column(db.Float)
    Lonely = db.Column(db.Float)
    Normal = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    diary = db.relationship('Diary')


views = Blueprint('views', __name__)

# Mendownload model dari bucket
def download_model():
    url = 'https://storage.googleapis.com/healthdiary-c23-ps111/model.h5'
    response = requests.get(url)
    with open('model.h5', 'wb') as f:
        f.write(response.content)

# Mendownload tokenizer dari bucket
def download_tokenizer():
    url = 'https://storage.googleapis.com/healthdiary-c23-ps111/tokenizer.pickle'
    response = requests.get(url)
    with open('tokenizer.pickle', 'wb') as f:
        f.write(response.content)

# Mendownload credentials.json dari bucket
def download_credentials():
    url = 'https://storage.googleapis.com/healthdiary-c23-ps111/credentials.json'
    response = requests.get(url)
    with open('credentials.json', 'wb') as f:
        f.write(response.content)

# Cek apakah file credentials.json sudah ada
if not os.path.exists('credentials.json'):
    download_credentials()

    # Memuat model dan tokenizer
def load_model_tokenizer():
    download_model()
    download_tokenizer()
    download_credentials()

    new_model = tf.keras.models.load_model('model.h5')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return new_model, tokenizer

def download_credentials():
    url = 'https://storage.googleapis.com/healthdiary-c23-ps111/credentials.json'
    response = requests.get(url)
    with open('credentials.json', 'wb') as f:
        f.write(response.content)

# Memuat model dan tokenizer
new_model, tokenizer = load_model_tokenizer()


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        diary = request.form.get('diary')

        if len(diary) < 1:
            flash('Your Diary is too short!', category='error') 
        else:
            
            # Fungsi untuk melakukan preprocessing pada teks
            en = spacy.load('en_core_web_sm')
            stopwords = en.Defaults.stop_words

            def preprocessing(sentence):
                # Convert text to lowercase, contraction, remove numbers, remove punctuation
                sentence = re.sub(r'\d+', '', contractions.fix(sentence.lower()).translate(str.maketrans('', '', string.punctuation)))

                # Remove stopwords
                text = [word for word in sentence.split() if word not in stopwords]

                # Remove extra whitespace
                sentence = ' '.join(text).strip()

                return sentence

            # Inisialisasi klien Cloud Translation
            translate_client = translate.Client()

            # Fungsi untuk menerjemahkan teks menggunakan Google Cloud Translation API
            def translate_text(text, target_language):
                translation = translate_client.translate(
                    text,
                    target_language=target_language
                )
                translated_text = translation['translatedText']
                return translated_text

            # Menerjemahkan teks input ke bahasa Inggris
            translated_text = translate_text(diary, 'en')

            # Cek hasil translate
            print(translated_text)

            # Melakukan preprocessing pada teks
            preprocessed_text = preprocessing(translated_text)

            # Cek hasil preprosessing
            print(preprocessed_text)

            # Prediksi menggunakan model dan tokenizer yang dimuat
            def predict_text_sentiment(seed_text, model, tokenizer):
                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                # Pad the sequences
                token_list = pad_sequences([token_list], padding='post')
                # Get the probabilities of predicting a word
                predicted = model.predict(token_list, verbose=0)[0]
                #print('Probabilitas:')
                #print('Anxiety : {:.2%}'.format(predicted[0]))
                #print('Depresi : {:.2%}'.format(predicted[1]))
                #print('Lonely : {:.2%}'.format(predicted[2]))
                #print('Normal : {:.2%}'.format(predicted[3]))
                return predicted

            # Menjalankan prediksi menggunakan model
            predict=predict_text_sentiment(preprocessed_text, new_model, tokenizer)
            anxiety=predict[0]
            depresi=predict[1]
            lonely=predict[2]
            normal=predict[3]

            new_diary = Diary(data=diary, Anxiety=anxiety*100, Depresi=depresi*100, Lonely=lonely*100, Normal=normal*100, user_id=current_user.id)
            datapredict=np.array(predict)
            predict_list=datapredict.tolist()
            db.session.add(new_diary) 
            db.session.commit()
            flash('Diary added!', category='success')
            return {'status': 'success', 'diary': diary, 'hasil_predict': predict_list}


    #return render_template("home.html", user=current_user)


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)