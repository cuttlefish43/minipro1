from django.urls import path
from . import views
urlpatterns = [
    path('',views.home, name ='home'),
    path('vote',views.vote, name='vote'),
    path('datasetDownload',views.datasetDownload,name='datasetDownload')
]