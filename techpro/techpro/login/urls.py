from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path('', views.sign, name='sign'),  # Entry point for login/signup page
    path('home/', views.home, name='home'),  # Home page after successful login/signup
    path('login/', views.login_view, name='login'),  # Handle login
    path('signup/', views.signup_view, name='signup'),  # Handle signup
    path('crop/', views.crop_view, name='crop_page'),
    path('fertilizer/', views.fertilizer_view, name='fertilizer_page'),
    path('croprec/', views.crop_recommendation, name='croprec'),
    path('fertrec/', views.fertilizer_recommendation, name='fertrec'),
]
