import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

data = pd.read_csv("Book2 (1).csv")

#data.head(50)

#data.info()
#data = data[:1000]

data = data.dropna()

data=data.drop(['Yield'], axis=1)

data=data.drop(['Year'], axis=1)

data=data.drop(['Production Units'], axis=1)

data=data.drop(['Area Units'], axis=1)

#data.info()

data.describe()

data.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

State = le.fit_transform(data.State)
District = le.fit_transform(data.District)
Crop = le.fit_transform(data.Crop)
Season = le.fit_transform(data.Season)
data['State'] = State
data['District'] = District
data['Crop'] = Crop
data['Season'] = Season

#data.head()

#data.head(30)

from sklearn.model_selection import train_test_split


x = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, r2_score
forest = RandomForestRegressor(n_estimators = 1000, criterion = "mse")
forest.fit(X_train,Y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
#print("MSE train: %.3f, test: %.3f" %(mean_squared_error(Y_train, y_train_pred), mean_squared_error(Y_test, y_test_pred)))
#print("R^2 train: %.3f, test: %.3f" %(r2_score(Y_train, y_train_pred), r2_score(Y_test, y_test_pred)))

#print(forest.score(X_test, Y_test))
forest.predict(X_test)
forest.predict([[1,4,3,1,630.0]])

#import matplotlib.pyplot as plt

# Data to plot
labels = 'gujarat', 'madhya pradesh', 'punjab', 'tamil nadu'
sizes = [130, 230, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice

# Plot
# plt.pie(sizes, explode=explode, labels=labels, colors=colors,
# autopct='%1.1f%%', shadow=True, startangle=140)
#
# plt.axis('equal')
#plt.show()

num=[]
for i in range(X_test.shape[0]):
    num.append(i+1)
# plt.scatter(num,y_test_pred,label="Predicted")
# plt.scatter(num,Y_test,label="Actual")
# plt.legend()
# plt.xlabel("Datapoints")
# plt.ylabel("Production")

def final(sta,dis,cro,seas,are):
    State = sta

    District= dis
    Crop = cro
    Season= seas
    Area= are
    out_1 = forest.predict([[float(State), float(Crop), float(District), float(Season),float(Area)]])
    return str(out_1[0])
    print(out_1)
    print('Crop production prediction ', out_1)

# from tkinter import *
# from tkinter import ttk
# from PIL import ImageTk, Image
# from tkinter import filedialog
# import os
# from io import BytesIO, StringIO
# import datetime


# def get_num(ine):
#     for letter in ine:
#         if letter.isdigit():
#             return int(letter) - 1


# root = Tk()
# root.geometry("492x670+100+50")
# root.title("Crop predictor")
# root.configure(bg="grey19")
# root.resizable(False, False)
# img = Image.open("apnaanaj.png")
# img = img.resize((256,128))
# photo1 = ImageTk.PhotoImage(img)
# label = Label(root, image=photo1)
# label.config(bg = "grey19")
# label.grid(padx = 120)


# def maini():
#     textbox1 = Label(root,text="Select to predict!")
#     textbox1.grid()
#     textbox1.config(font=("kenyan coffee", 11, "bold"),fg="red",bg = "grey19")
#     textbox1.place(relx = 0.362,rely = 0.23)

#     textbox2 = Label(root, text="Enter State Name")
#     textbox2.grid()
#     textbox2.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#     textbox2.place(relx=0.080, rely=0.30)

#     textbox3 = Label(root, text="Enter District Name")
#     textbox3.grid()
#     textbox3.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#     textbox3.place(relx=0.080, rely=0.37)

#     textbox4 = Label(root, text="Enter a Crop")
#     textbox4.grid()
#     textbox4.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#     textbox4.place(relx=0.080, rely=0.44)

#     textbox5 = Label(root, text="Enter Cultivating Season")
#     textbox5.grid()
#     textbox5.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#     textbox5.place(relx=0.080, rely=0.51)

#     textbox6 = Label(root, text="Enter Cultivating area (hectare)")
#     textbox6.grid()
#     textbox6.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#     textbox6.place(relx=0.080, rely=0.58)


#     select = ttk.Combobox(root)
#     select['values']=["Gujrat","Madhya Pradesh","Punjab","Tamil Nadu"]
#     select.config(font=("kenyan coffee", 11,"bold"), background= "grey12")

#     select.grid(padx=100, pady=0)
#     select.place(rely=0.30, relx=0.548)
#     select.current(0)

#     select1 = ttk.Combobox(root, values=["Ahmedabad","Ashoknagar","Betul","Coimbatore","Damoh","Fazilka","Gandhinagar","Jabalpur","Jamnagar","Khargone","Ludhiana","Madhurai","Nagapattinam","Pathankot","Patiala","Rajkot","Salem","Surat","Theni","Thoothukudi","Trippur","Vellore","Vidisha","Virudhnagar"])
#     select1.config(font=("kenyan coffee", 11, "bold"))
#     select1.grid(padx=100, pady=0)
#     select1.place(rely=0.37, relx=0.548)
#     select1.current(0)


#     select2 = ttk.Combobox(root, values=["Arhar", "Bajra", "Gram", "Maize","Rice","Wheat"])
#     select2.config(font=("kenyan coffee", 11, "bold"))
#     select2.grid(padx=100, pady=0)
#     select2.place(rely=0.44, relx=0.548)
#     select2.current(0)


#     select3 = ttk.Combobox(root, values=["Winter","Kharif","Rabi","Summer","Autumn"])
#     select3.config(font=("kenyan coffee", 11, "bold"))
#     select3.grid(padx=100, pady=0)
#     select3.place(rely=0.51, relx=0.548)
#     select3.current(0)


#     select4 = ttk.Entry(root,width = 30)
#     select4.grid(padx=100, pady=0)
#     select4.place(rely=0.58, relx=0.548)

#     def on_click():
#         state_input = str(select.current())
#         district_input = str(select1.current())
#         crop_input = str(select2.current())
#         season_input = str(select3.current())
#         area = select4.get()

#         textbox6 = Label(root, text="Crop Production Prediction: "+final(state_input,district_input,crop_input,season_input,area))
#         textbox6.grid()
#         textbox6.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="grey19")
#         textbox6.place(relx=0.215, rely=0.80)


#     submit_button = Button(root,text=" Predict ",command = on_click)
#     submit_button.grid()
#     submit_button.config(font=("kenyan coffee", 11, "bold"), fg="white", bg="red")
#     submit_button.place(relx=0.430, rely=0.69)

# maini()
# mainloop()
final(1,1,1,1,65)

from flask import Flask, render_template,request
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)

ENV = 'dev'
if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:hr@localhost/Apna Anaaj'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class project(db.Model):
    __tablename__ = 'anaaj'
    states_name = db.Column(db.String(200), unique=True)
    district_name = db.Column(db.String(200), unique=True)
    crop_name = db.Column(db.String(200), unique=True)
    season_name = db.Column(db.String(200), unique=True)
    area = db.Column(db.Integer(), primary_key = True)

    def __init__(self, states_name, district_name, crop_name, season_name, area):
        self.states_name = states_name
        self.district_name = district_name
        self.crop_name = crop_name
        self.season_name = season_name
        self.area = area



@app.route("/")
def index():
    return render_template("Home.html",result = None)

@app.route("/home")
def home():
    return render_template("Home.html",result = None)

@app.route("/about")
def about():
    return render_template("About.html")


ENV = 'dev'
if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:hr@localhost/Apna Anaaj'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class project(db.Model):
    __tablename__ = 'anaaj'
    states_name = db.Column(db.String(200),primary_key = True)
    district_name = db.Column(db.String(200))
    crop_name = db.Column(db.String(200))
    season_name = db.Column(db.String(200))
    area = db.Column(db.Integer())
    f_res= db.Column(db.DECIMAL())

    def __init__(self, states_name, district_name, crop_name, season_name, area,f_res):
        self.states_name = states_name
        self.district_name = district_name
        self.crop_name = crop_name
        self.season_name = season_name
        self.area = area
        self.f_res = f_res

@app.route('/get', methods=["GET", "POST"])
def get():
    states_name = ["Gujrat","Madhya Pradesh","Punjab","Tamil Nadu"]
    district_name = ["Ahmedabad","Ashoknagar","Betul","Coimbatore","Damoh","Fazilka","Gandhinagar","Jabalpur","Jamnagar","Khargone","Ludhiana","Madhurai","Nagapattinam","Pathankot","Patiala","Rajkot","Salem","Surat","Theni","Thoothukudi","Trippur","Vellore","Vidisha","Virudhnagar"]
    crop_name = ["Arhar", "Bajra", "Gram", "Maize","Rice","Wheat"]
    season_name = ["Winter","Kharif","Rabi","Summer","Autumn"]
    final_result = None
    if request.method == "POST":
        real_state_name = request.form.get("state")
        real_district_name = request.form.get("district")
        real_crop_name = request.form.get("crop")
        real_season_name = request.form.get("season")
        state = states_name.index(request.form.get("state"))+1
        district = district_name.index(request.form.get("district"))+1
        crop = crop_name.index(request.form.get("crop"))+1
        season = season_name.index(request.form.get("season"))+1
        area = request.form.get("area")
        final_result = final(state,crop,district,season,area)
        f = open("temp.txt","w")
        f.write(final_result)
        f.close()
        if db.session.query(project).filter(project.states_name == real_state_name).count() == 0:
            data = project(real_state_name, real_district_name, real_crop_name, real_season_name, area,final_result)
            db.session.add(data)
            db.session.commit()
            print("done")

    return render_template("Home.html",result = final_result)


@app.route('/show')
def show():
    f = open('temp.txt','r+')
    final = f.read()
    print("dd",final)

    return render_template("Home.html", result=final)


if __name__ == "__main__":
    app.run(debug = True)