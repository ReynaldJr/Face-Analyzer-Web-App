from django.urls import path
from faceapp import views
urlpatterns = [
    path('',views.index_three, name='index2')
]
