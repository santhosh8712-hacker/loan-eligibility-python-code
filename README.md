import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import sqlite3
import warnings
from werkzeug.security import generate_password_hash, check_password_hash
from pytesseract import pytesseract
import cv2
import re



import requests 


path_to_tesseract = "C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = '123'  

import sqlite3
database = 'new.db'

def init_db():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            aadhar TEXT UNIQUE,
            password TEXT,
            gender TEXT,
            marital_status TEXT,
            education TEXT,
            self_employed TEXT,
            income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term INTEGER,
            credit_history INTEGER,
            property_area TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS education (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            aadhar TEXT UNIQUE,
            password TEXT,
            applicantIncome REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term INTEGER,
            credit_history INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Run this function once to update the database
init_db()


# Load Model
filename = 'model/model.pkl'
model = pickle.load(open(filename, 'rb'))

filename1 = 'model2.pkl'
model2 = pickle.load(open(filename1, 'rb'))

url = "http://100.85.31.171:8080/shot.jpg"

def clean_and_extract_aadhaar(ocr_text):
    cleaned_text = ocr_text.replace('\n', ' ')  
    cleaned_text = re.sub(r'[^0-9\s]', '', cleaned_text)  
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'  
    matches = re.findall(aadhaar_pattern, cleaned_text)
    #print(cleaned_text)
    if matches:
        aadhaar_number = matches[0].replace(" ", "")  # Remove spaces
        return aadhaar_number
    return None



def update_profile_data(email, aadhar, gender, marital_status, education, self_employed, income, coapplicant_income, loan_amount, loan_term, credit_history, property_area):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Check if the user exists by email or Aadhar
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()

    if user:
        # Update existing user profile
        cursor.execute('''
            UPDATE users 
            SET gender = ?, aadhar=?, marital_status = ?, education = ?, self_employed=?, income = ?, 
                coapplicant_income = ?, loan_amount = ?, loan_term = ?, 
                credit_history = ?, property_area = ?
            WHERE email = ?
        ''', (gender, aadhar, marital_status, education, self_employed, income, coapplicant_income, loan_amount, loan_term, credit_history, property_area, email))

        conn.commit()
        conn.close()
        return "Profile Updated Successfully"
    else:
        conn.close()
        return "User Not Found"

def update_data(email, aadhar, applicantIncome, coapplicant_income, loan_amount, loan_term, credit_history):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM education WHERE email = ?", (email,))
    user = cursor.fetchone()

    if user:
        cursor.execute('''
            UPDATE education 
            SET aadhar=?, applicantIncome=?, coapplicant_income=?, loan_amount=?, loan_term=?, credit_history=?
            WHERE email=?
        ''', (aadhar, applicantIncome, coapplicant_income, loan_amount, loan_term, credit_history, email))

        conn.commit()
        conn.close()
        return "Profile Updated Successfully"
    else:
        conn.close()
        return "User Not Found"

    

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        
        # Check login in 'users' table
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password,))
        user = cursor.fetchone()
        
        if user:
            session['email'] = email

            username = user[1]  # assuming username is the second column in 'users' table

            # Check if user already exists in 'education' table
            cursor.execute("SELECT * FROM education WHERE email = ?", (email,))
            edu = cursor.fetchone()

            if not edu:
                # Insert into education table with username and password
                cursor.execute('''
                    INSERT INTO education (username, email, aadhar, password)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, '', password))  # aadhar left blank, will be updated later

                conn.commit()
            
            conn.close()
            return render_template('index.html')
        else:
            conn.close()
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            conn = sqlite3.connect(database)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username or email already exists')
    
    return render_template('register.html')


@app.route('/update_profile', methods=['POST'])
def update_profile():
    email = session.get('email')  # Corrected key name
    if not email:
        return jsonify({"message": "User not logged in"}), 401

    aadhar = request.form['aadhar']
    gender = request.form['Gender']
    marital_status = request.form['Married']
    education = request.form['Education']
    self_employed = request.form['Self_Employed']
    income = request.form['ApplicantIncome']
    coapplicant_income = request.form['CoapplicantIncome']
    loan_amount = request.form['LoanAmount']
    loan_term = request.form['Loan_Amount_Term']
    credit_history = request.form['Credit_History']
    property_area = request.form['Property_Area']

    result = update_profile_data(email, aadhar, gender, marital_status, education, self_employed,
                                 income, coapplicant_income, loan_amount, loan_term, credit_history, property_area)

    return render_template('index.html')
@app.route('/update_profile1', methods=['GET','POST'])
def update_profile1():
    email = session.get('email')
    if not email:
        return jsonify({"message": "User not logged in"}), 401

    try:
        # Match keys exactly with HTML input 'name' attributes
        aadhar = request.form['aadhar']
        applicant_income = request.form['ApplicantIncome']
        coapplicant_income = request.form['CoapplicantIncome']
        loan_amount = request.form['LoanAmount']
        loan_term = request.form['Loan_Amount_Term']
        credit_history = request.form['Credit_History']

        result1 = update_data(email, aadhar, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history)
        return render_template('index.html')
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 400




def capture_aadhaar():
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (1000, 600))  
    
    # Extract text using OCR
    text = pytesseract.image_to_string(img, config='--psm 6')
    aadhaar_number = clean_and_extract_aadhaar(text)

    cv2.imshow("Captured Frame", img)
    cv2.waitKey(10000)  # Show frame for 2 seconds
    cv2.destroyAllWindows()  # Close the camera window automatically


    if aadhaar_number:
        #print(aadhaar_number)
        return aadhaar_number
    else:
        return None

@app.route('/aadhar', methods=['GET', 'POST'])
def aadhar():
    aadhar = capture_aadhaar()
    print(aadhar)    
    if not aadhar:
        return "Aadhaar number not found!"

    conn = sqlite3.connect(database)
    cur = conn.cursor()
    
    # Fetch user details based on Aadhaar number
    cur.execute("SELECT email, aadhar, gender, income, coapplicant_income, loan_amount FROM users WHERE aadhar = ?", (aadhar,))
    user_data = cur.fetchone()
    
    if not user_data:
        return render_template("index.html")

    email, aadhar, gender, income, coapplicant_income, loan_amount = user_data
    print(user_data)

    # Fetch other required data for prediction
    cur.execute("SELECT gender, marital_status, education, self_employed, income, coapplicant_income, loan_amount, loan_term, credit_history, property_area FROM users WHERE aadhar = ?", (aadhar,))
    data = cur.fetchone()
    
    conn.close()

    if data:
        int_features = [float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in data]
        final_features = np.array([int_features], dtype=np.float32)

        prediction = model.predict(final_features)
        output = round(float(prediction[0]), 2)

        if output == 0:
            result_text = 'LOAN APPROVAL ❌'
        else:
            result_text = 'LOAN APPROVAL ✅'
        
        return render_template(
            'result.html', 
            prediction_text=result_text, 
            email=email, 
            aadhar=aadhar, 
            gender=gender, 
            income=income, 
            coapplicant_income=coapplicant_income, 
            loan_amount=loan_amount
        )

    return render_template("index.html")

    
@app.route('/aadhar1', methods=['GET', 'POST'])
def aadhar1():
    aadhar = capture_aadhaar()
    print(aadhar)    
    if not aadhar:
        return "Aadhaar number not found!"

    conn = sqlite3.connect(database)
    cur = conn.cursor()
    
    # Fetch user details based on Aadhaar number
    cur.execute("SELECT email, aadhar, applicantIncome, coapplicant_income, loan_amount FROM education WHERE aadhar = ?", (aadhar,))
    user_data = cur.fetchone()
    
    if not user_data:
        return render_template("index.html")

    email, aadhar, applicant_income, coapplicant_income, loan_amount = user_data
    print(user_data)

    # Fetch data for prediction
    cur.execute("SELECT applicantIncome, coapplicant_income, loan_amount, loan_term, credit_history FROM education WHERE aadhar = ?", (aadhar,))
    data = cur.fetchone()
    
    conn.close()

    if data:
        int_features = [float(x) for x in data]
        final_features = np.array([int_features], dtype=np.float32)

        prediction = model2.predict(final_features)
        output = round(float(prediction[0]), 2)

        result_text = 'LOAN APPROVAL ✅' if output == 1 else 'LOAN APPROVAL ❌'
        
        return render_template(
            'result1.html', 
            prediction_text=result_text, 
            email=email, 
            aadhar=aadhar, 
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income, 
            loan_amount=loan_amount
        )

    return render_template("index.html")
 

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/profile', methods=['GET','POST'])
def profile():    
    return render_template('profile.html')

@app.route('/profile1', methods=['GET','POST'])
def profile1():    
    return render_template('profile1.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([int_features])  
    prediction = model.predict(final_features)
    output = round(float(prediction[0]), 2)  

    if output == 0:
        result_text = 'LOAN APPROVAL ❌'
    else:
        result_text = 'LOAN APPROVAL ✅'
    
    return render_template('result.html', prediction_text=result_text)



@app.route('/results',methods=['POST'])
def results1():

    data = request.get_json(force=True)
    prediction = model2.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




if __name__ == "__main__":
    app.run(debug=False, port=800)
