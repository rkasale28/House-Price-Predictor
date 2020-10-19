from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/',views.handle_iteration, name="handle_iteration")
]
