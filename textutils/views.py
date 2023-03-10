from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
import pandas as pd   
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def index(request):
    return render(request, 'index.html')

def analyze(request):
    crop = pd.read_csv("dataset/Crop_recommendation.csv")

    crop.drop_duplicates()
    attr=["N","P","K","temperature","humidity","rainfall","label"]
    if crop.isna().any().sum() !=0:
        for i in range(len(attr)):
            crop[atrr[i]].fillna(0.0, inplace = True)

    crop.columns = crop.columns.str.replace(' ', '') 
    features = crop[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]

    target = crop['label']

    x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)
    
    RF = RandomForestClassifier(n_estimators=20, random_state=0)

    RF.fit(x_train,y_train)
    N = request.POST.get('nitrogen', 'default')
    P = request.POST.get('phosphorous', 'default')
    K = request.POST.get('potassium', 'default')
    temp = request.POST.get('temperature', 'default')
    humidity = request.POST.get('humidity', 'default')
    ph =request.POST.get('ph', 'default')
    rainfall = request.POST.get('rainfall', 'default')

    userInput = [N, P, K, temp, humidity, ph, rainfall]
    
    result = RF.predict([userInput])[0]

    params = {'purpose':'Predicted Crop: ', 'analyzed_text': result.upper()}
    return render(request, 'analyze.html', params)    
    
def about_us(request):
    return render(request, 'About_us.html')    




