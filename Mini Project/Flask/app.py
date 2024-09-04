from flask import Flask, render_template,request,redirect,url_for, flash
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import joblib
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from PIL import Imagepi
import cv2



app = Flask("__name__")
app.secret_key = "batch-10"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "mini project"
app.config["UPLOAD_FOLDER"] = 'uploads/'
app.config["ALLOWED_EXTENSIONS"] = {'png','jpg','jpeg','gif'}

model = keras.models.load_model('Alex - Net.h5')
feature_extractor = tf.keras.models.Model(inputs = model.input,outputs=model.layers[-6].output)
ann_model = tf.keras.models.load_model('best_model_final_model.h5')


mysql = MySQL(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    processed_images = []
    img = cv2.imread(image_path)
    # Resize image to (227, 227)
    resized_img = cv2.resize(img, (227, 227))
    # Preprocess the image (normalize pixel values to range [0, 1])
    resized_img = resized_img / 255.0  
    processed_images.append(resized_img)
    return np.array(processed_images)


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        joint_pain = request.form["Joint pain"]
        Gender = request.form["gender"]
        Age = request.form["Age"]
        
        Menopause_age = request.form["Menopause Age"]
        
        height = request.form["Height"]
        
        weight = request.form["Weight"]
        
        smoker = request.form["Smoker"]
        diabetic = request.form["Diabetic"]
        hypothyroidism = request.form["Hypothyroidism"]
        num_of_pregnanacies = request.form["Number of Pregnancies"]
        
        seizer_disorder = request.form["Seizer Disorder"]
        estrogen_use = request.form["Estrogen Use"]
        occupation = request.form["Occupation"]
        
        history_of_fracture = request.form["History of Fracture"]
        
        dialysis = request.form["Dialysis"]
        family_history_of_fracture = request.form["Family History of Osteoporosis"]
        max_walking_distance = request.form["Maximum Walking distance"]
        
        daily_eating_habits = request.form["Daily Eating habits"]
        
        medical_history = request.form["Medical History"]
        
        t_score = request.form["T-score Value"]
        
        z_score = request.form["Z-Score Value"]
        
        bmi = request.form["BMI"]
       
        obesity = request.form["Obesity"]
        
    if 'file' not in request.files:
        return redirect(url_for('detect'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename) and request.method=="POST":
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    joint_pain = 1 if joint_pain.lower() == 'yes' else 0
    Gender = 1 if Gender.lower()=='male' else 0
    smoker = 1 if smoker.lower() == 'yes' else 0
    diabetic = 1 if diabetic.lower() == 'yes' else 0
    hypothyroidism = 1 if hypothyroidism.lower() == 'yes' else 0
    seizer_disorder = 1 if seizer_disorder.lower() == 'yes' else 0
    estrogen_use = 1 if estrogen_use.lower() == 'yes' else 0
    dialysis = 1 if dialysis.lower() == 'yes' else 0
    family_history_of_fracture = 1 if family_history_of_fracture.lower() == 'yes' else 0

    with open('occ.pkl', 'rb') as f:
        occ = pickle.load(f)
        occupation = occ.transform([occupation])
    with open('hof.pkl', 'rb') as f:
        hof = pickle.load(f)
        history_of_fracture = hof.transform([history_of_fracture])
    with open('deh.pkl', 'rb') as f:
        deh = pickle.load(f)
        daily_eating_habits = deh.transform([daily_eating_habits])
    with open('mh.pkl', 'rb') as f:
        mh = pickle.load(f)
        medical_history = mh.transform([medical_history])
    if obesity.lower() == 'normal weight':
        obesity = 0
    elif obesity.lower() == 'obesity':
        obesity = 1
    elif obesity.lower() == 'over weight':
        obesity = 2
    elif obesity.lower() == 'overweight':
        obesity = 3
    else:
        obesity = 4

    Age = float((float(Age)-17)/(90))
    Menopause_age = float(float(Menopause_age)/(57))
    height = float((float(height)-1.373)/(1.8288-1.373))
    weight = float((float(weight)-39)/(98-39))
    num_of_pregnanacies = float(float(num_of_pregnanacies)/(7))
    occupation = float(float(occupation)/(21))
    history_of_fracture = float(float(history_of_fracture)/46)
    max_walking_distance = float(float(max_walking_distance)/10)
    daily_eating_habits = float(float(daily_eating_habits)/ 25)
    medical_history = float(float(medical_history)/81)
    t_score = float((float(t_score)+2.99)/(-0.16+2.99))
    z_score = float((float(z_score)+2.99)/(0.73+2.99))
    bmi = float((float(bmi)-16.139)/(42.75438-16.139))
    obesity = float(float(obesity)/4)


    numeric = np.array([joint_pain,Gender,Age,Menopause_age,height,weight,
                            smoker,diabetic,hypothyroidism,num_of_pregnanacies,
                            seizer_disorder,estrogen_use,occupation,history_of_fracture,
                            dialysis,family_history_of_fracture,max_walking_distance,
                            daily_eating_habits,medical_history,t_score,z_score,bmi,obesity])
    numeric_2d = numeric.reshape(1, -1)  


    img_array = preprocess_image(file_path)
    img_features = feature_extractor.predict(img_array)
    img_features = np.array(img_features, dtype=np.float32)
    min_val = np.min(img_features)
    max_val = np.max(img_features)
    normalized_features = (img_features - min_val) / (max_val - min_val)
    
    x = np.concatenate((normalized_features,numeric_2d),axis=1)
    output = ann_model.predict(x)
    predicted_index = np.argmax(output, axis=1)
    label_encoder = joblib.load('label_encoder.pkl')
    predicted_label = label_encoder.inverse_transform(predicted_index)
    result = predicted_label[0]
    return render_template('result.html',output = result)

        

@login_manager.user_loader
def load_user(user_id):
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1])
    return None



@app.route('/')
@app.route('/home')
@app.route('/index')
def index():
    return render_template('index.html')
    
@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()
        cursor.close()
        if user_data and user_data[2] == password:
            user = User(id=user_data[0], username=user_data[1])
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users(name, username, password) VALUES (%s, %s, %s)",(name, username,password))
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
    

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)