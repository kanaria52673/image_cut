# -*- coding: utf-8 -*-
from flask import Flask
from flask import render_template, request, redirect,flash,url_for
from flask_sqlalchemy import SQLAlchemy
import os
from flask_login import UserMixin, LoginManager,login_user,logout_user, login_required,current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
from flask_wtf.csrf import CSRFProtect
import numpy as np
import cv2
import zipfile
import shutil
from PIL import Image
import requests
import json
import io
from pathlib import Path

# ハッシュ化の値を変更
with open("secret.json") as f:
  secret_json = json.load(f)
app = Flask(__name__, static_url_path="")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = secret_json["secret_key"]

# 制限を記載
limiter = Limiter(app, key_func=get_remote_address, default_limits=["50 per minute"])
db = SQLAlchemy(app)
csrf = CSRFProtect(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(50), nullable=False, unique=True)
  password = db.Column(db.String(25),nullable=False)


shutil.rmtree("./static/")
IMG_DIR = "./static/"

BASE_DIR = os.path.dirname(__file__)
IMG_PATH = BASE_DIR + IMG_DIR

if not os.path.isdir(IMG_DIR):
    os.mkdir(IMG_DIR)
    

def aspect_ratio(ax,ay):
    x, y = ax, ay
    while y:
        x, y = y, x % y
    return ax/x, ay/x


def faceDetectionFromPath(img,num,f_direction,secret_json=secret_json):

    count = 0
    count_1 = 0
    face_api_url = secret_json['face_api_url']
    subscription_key =  secret_json['subscription_key']
    assert subscription_key
    cvImg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        binary_img = output.getvalue()

    headers = {
        'Content-Type' : 'application/octet-stream',
        'Ocp-Apim-Subscription-Key':subscription_key
    }

    params = {'returnFaceId': 'true',
    'returnFaceAttributes': 'age, gender, headPose, smile, facialHair, glasses, emotion, hair, makeup, occlusion, accessories, blur, exposure, noise'
    }
    res = requests.post(face_api_url,params=params, headers=headers, data =   binary_img  )
    results = res.json()
    # 目を認識させたほうが精度が高くなったため
    cascade_path = "./lib/haarcascade_eye_tree_eyeglasses.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    f=True
    for result in results:
      try:
        rect = result['faceRectangle']
        if f_direction:
          count += (rect['left']+rect['left']+rect['width']) / 2
        else:
          count += (rect['top']+rect['top']+rect['height']) / 2
        count_1 += 1
        f = False
      except:
        pass
    if f:
      for rect in facerect:
        if f_direction:
          count += (rect[0]+rect[0]+rect[2]) / 2
        else:
          count += (rect[1]+rect[1]+rect[3]) / 2
        count_1 += 1


    try:
      count /= count_1
    except ZeroDivisionError:
      count = 0
    if f:
      print(f"{num}番目は openCV算出 中央の位置 {count}")
    else:
      print(f"{num}番目は Azure算出 中央の位置 {count}")
    return count

def trim(im,fa_cut,w,h,f_direction):
  im_c = np.array(im)
  # サイズ比を取得
  w1,h1 = aspect_ratio(w,h)
  left = top = 0
  # 画像の回転(縦画像用)
  if f_direction: 
    im_c = np.rot90(im_c, 2)
  
  right = im_c.shape[1]
  bottom = h1 * (right / w1)  
  if bottom > im_c.shape[0] and fa_cut == 0:
    bottom = im_c.shape[0]
    right = w1 * (bottom / h1)
    left = (im_c.shape[1] - right) /2 
    right += left
      
  elif bottom > im_c.shape[0]:
    bottom = im_c.shape[0]
    right = w1 * (bottom / h1)
    left = (im_c.shape[1] - right) /2 
    right += left
    fa_cut -= (left + right) /2
    left += fa_cut
    right += fa_cut
    if left < 0:
      right += -1 * left
      left = 0
  elif fa_cut == 0:
    pass
  else:
    fa_cut -= (top + bottom) /2
    top += fa_cut
    bottom += fa_cut
    if top >= 0 and bottom <= im_c.shape[0]:
      pass
    elif top >= 0:
      fa_cut = bottom - im_c.shape[0]
      bottom -= fa_cut
      top -= fa_cut
    else:
      bottom += -1 * top
      top = 0
  
  del im_c
  # error回避用
  im_trimmed = im.crop((left,top,right,bottom))
  return im_trimmed


@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    img_name = ""
    
    z_pass = ""

    if request.method == 'POST':
        IMG_DIR = f"./static/{current_user.username}/"
        if not os.path.isdir(IMG_DIR):
            os.mkdir(IMG_DIR)
        shutil.rmtree(IMG_DIR)
        if not os.path.isdir(IMG_DIR):
            os.mkdir(IMG_DIR)
        if not os.path.isdir(IMG_DIR + "im/"):
            os.mkdir(IMG_DIR + "im/")
        if not os.path.isdir(IMG_DIR + "images/"):
            os.mkdir(IMG_DIR + "images/")
        if not os.path.isdir(IMG_DIR + "sample/"):
            os.mkdir(IMG_DIR + "sample/")

        # 画像の切り取りサイズを指定
        wi = int(request.form.get('wi1'))
        he = int(request.form.get('he1'))
        f_direction = False
        if wi < he:
          f_direction = True
          
        num = 0
        stream = request.files['image']
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        st = request.files['sd']
        
        if st:
          BASE_DIR = Path(__file__).resolve().parent
          IMG_PATH = f"{BASE_DIR}/static/{current_user.username}/im/"
          with zipfile.ZipFile(st, 'r')as f:
            f.extractall(IMG_PATH)
          files = os.listdir(IMG_PATH)
          # ディレクトリトラバーサル用（意味ある？）
          files = [name.split(".")[-2:] for name in files if name.split(".")[-1] in ["png","jpg","jpeg"]]
          
          
          for val in files:
            try:
              num += 1
              val = val[0] + '.' + val[1]
                      
              IMG_PATH_1 = IMG_PATH + val
              img = Image.open(IMG_PATH_1)
              img = np.array(img)
                      
              #画像から顔の中央値を取得
              fa_cut = faceDetectionFromPath(img,num,f_direction)
              #画像の形式を変更
              im = Image.fromarray(img)
              #　画像切り取り
              img = trim(im,fa_cut,wi,he,f_direction)
                      
              # 画像の名前を指定(数字の連番)
              img_name = format(num,'06') + ".jpg"
                      
              img_name = f"./static/{current_user.username}/sample/{img_name}"
              img.save(img_name)
            except:
              print(f'error {num } 番目') 
          z_pass = f"./static/{current_user.username}/sample"
          shutil.make_archive(z_pass, 'zip', root_dir=z_pass)
          z_pass = f"./{current_user.username}/sample.zip"
          
        # 単体の画像があるか確認
        if len(img_array) != 0:
            img = cv2.imdecode(img_array, 1)
            # サーバの制限回避用
            while img.shape[0] * img.shape[1] > 7000000:
              img = cv2.resize(img, dsize=(img.shape[1]//50*49, img.shape[0]//50*49))

            num += 1
            # 画像から顔の中央値を取得
            fa_cut = faceDetectionFromPath(img,num,f_direction)
            im = Image.fromarray(img)
            # 画像の名前を指定(数字の連番)
            im = trim(im,fa_cut,wi,he,f_direction)
            # 画像の名前を指定(数字の連番)
            img_name = format(num,'06') + ".jpg"
            w = f"static/{current_user.username}/images/{img_name}"
            im = np.array(im)
            # 画像の保存
            cv2.imwrite(w, im)
            img_name = f"{current_user.username}/images/{img_name}"
    return render_template('index.html', img_name=img_name,z_pass=z_pass)

# アカウント作成
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        if 25 >= len(password) >= 8 and 50 >= len(username) >= 10:
          
                try:
                  user = User(username=username, password=generate_password_hash(password, method='sha256'))
                  db.session.add(user)
                  db.session.commit()
                  flash('アカウント作成に成功しました。')
                  return redirect('/login')
                except:
                  db.session.rollback()
                  flash('アカウント作成に失敗しました。他ユーザとユーザ名が重複しております。')
                  return redirect(url_for('signup'))

        else:
            flash('アカウント作成に失敗しました。入力要件を満たしていません。')
            return redirect(url_for('signup'))
    else:
        flash('アカウントを作成してください。')
        return render_template('signup.html')


# ログイン
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("15 per hour")
def login():
    if request.method == "POST":
        username = request.form.get('username')
        if len(username) > 50 or len(username) < 10:
          flash('ユーザ名は10~50文字以内でお願いいたします。')
          return redirect(url_for('login'))
        password = request.form.get('password')
        if len(password) > 25 or len(password) < 8:
          flash('パスワードは8~25文字以内でお願いいたします。')
          return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if check_password_hash(user.password, password):
          login_user(user)
          return redirect('/')
        elif user:
          flash('パスワードが間違っています。')
          return redirect(url_for('login'))
        else:
          flash('存在しないユーザです。')
          return redirect(url_for('login'))

    else:
        flash('ログインしてください。')
        return render_template('login.html')

# ログアウト
@app.route('/logout')
@login_required
def logout():
  # 作成したデータも削除
  shutil.rmtree(f"./static/{current_user.username}/")
  logout_user()
  return redirect('/login')



# アカウント削除
@app.route('/profile/delete/', methods=['GET', 'POST'])
@login_required
def profile_delete():
    shutil.rmtree(f"./static/{current_user.username}/")
    kura = User.query.filter_by(id=current_user.id).first()
    db.session.delete(kura) 
    db.session.commit()
    return redirect('/')

# 利用規約
@app.route('/terms_of_service', methods=['GET'])
def terms_of_service():
    return render_template('terms_of_service.html')

if __name__ == '__main__':
    app.run()
