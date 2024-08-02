from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from django.contrib.auth import get_user_model, authenticate, login as auth_login
from django.contrib import messages

# Load crop recommendation dataset
crop_file_path = 'login/Crop_recommendation.csv'
crop_data = pd.read_csv(crop_file_path)

crop_features = ['Nitrate', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'ph', 'Rainfall']
X_crop = crop_data[crop_features]
y_crop = crop_data['Label']

crop_label_encoder = LabelEncoder()
y_crop = crop_label_encoder.fit_transform(y_crop)

crop_scaler = StandardScaler()
X_crop = crop_scaler.fit_transform(X_crop)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop, y_crop)

def crop_recommendation(request):
    predicted_crop = None

    if request.method == 'POST':
        nitrate = float(request.POST.get('nitrate'))
        phosphorus = float(request.POST.get('phosphorus'))
        potassium = float(request.POST.get('potassium'))
        temperature = float(request.POST.get('temperature'))
        humidity = float(request.POST.get('humidity'))
        ph = float(request.POST.get('ph'))
        rainfall = float(request.POST.get('rainfall'))

        input_data = np.array([[nitrate, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        input_data_scaled = crop_scaler.transform(input_data)
        
        prediction = crop_model.predict(input_data_scaled)
        predicted_crop = crop_label_encoder.inverse_transform(prediction)[0]

    return render(request, 'crop.html', {'predicted_crop': predicted_crop})

# Load fertilizer recommendation dataset
fertilizer_file_path = 'login/Fertilizer_Prediction.csv'
fertilizer_data = pd.read_csv(fertilizer_file_path)

fertilizer_features = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
X_fertilizer = fertilizer_data[fertilizer_features]
y_fertilizer = fertilizer_data['Fertilizer Name']

soil_label_encoder = LabelEncoder()
X_fertilizer['Soil Type'] = soil_label_encoder.fit_transform(X_fertilizer['Soil Type'])

crop_label_encoder = LabelEncoder()
X_fertilizer['Crop Type'] = crop_label_encoder.fit_transform(X_fertilizer['Crop Type'])

fertilizer_label_encoder = LabelEncoder()
y_fertilizer = fertilizer_label_encoder.fit_transform(y_fertilizer)

X_fertilizer_train, X_fertilizer_test, y_fertilizer_train, y_fertilizer_test = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

fertilizer_scaler = StandardScaler()
X_fertilizer_train = fertilizer_scaler.fit_transform(X_fertilizer_train)
X_fertilizer_test = fertilizer_scaler.transform(X_fertilizer_test)

fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fertilizer_train, y_fertilizer_train)

def fertilizer_recommendation(request):
    predicted_fertilizer = None

    if request.method == 'POST':
        temperature = float(request.POST.get('temperature'))
        humidity = float(request.POST.get('humidity'))
        moisture = float(request.POST.get('moisture'))
        soil_type = request.POST.get('soil_type')
        crop_type = request.POST.get('crop_type')
        nitrogen = float(request.POST.get('nitrogen'))
        potassium = float(request.POST.get('potassium'))
        phosphorus = float(request.POST.get('phosphorus'))

        try:
            soil_type_encoded = soil_label_encoder.transform([soil_type])[0]
            crop_type_encoded = crop_label_encoder.transform([crop_type])[0]

            input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorus]])
            input_data_scaled = fertilizer_scaler.transform(input_data)
            prediction = fertilizer_model.predict(input_data_scaled)
            predicted_fertilizer = fertilizer_label_encoder.inverse_transform(prediction)[0]
        except ValueError as e:
            predicted_fertilizer = str(e)

    return render(request, 'fertilizer.html', {'predicted_fertilizer': predicted_fertilizer})

# User authentication views
User = get_user_model()

def sign(request):
    return render(request, 'login_signup.html')

def crop_view(request):
    return render(request, 'crop.html')

def fertilizer_view(request):
    return render(request, 'fertilizer.html')

def home(request):
    return render(request, 'home.html')

def signup_view(request):
    warning_message = None
    if request.method == 'POST':
        email = request.POST['new_username']
        password1 = request.POST['new_password']
        password2 = request.POST['confirm_password']

        if password1 == password2:
            if User.objects.filter(email=email).exists():
                warning_message = "Email already taken"
            else:
                user = User.objects.create_user(email=email, password=password1)
                user.save()
                return redirect('login')
        else:
            warning_message = "Passwords do not match"

    return render(request, 'login_signup.html', {'warning_message': warning_message})

def login_view(request):
    warning_message = None
    if request.method == 'POST':
        email1 = request.POST['username']
        password1 = request.POST['password']
        user = authenticate(request, email=email1, password=password1)
        if user is not None:
            auth_login(request, user)
            return redirect('home')
        else:
            warning_message = "Invalid credentials"

    return render(request, 'login_signup.html', {'warning_message': warning_message})
