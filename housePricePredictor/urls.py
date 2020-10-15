from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('intermediate',views.intermediate,name='intermediate'),
    path('result',views.result,name='result')
]
