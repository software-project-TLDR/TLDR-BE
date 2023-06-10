from django.urls import path
from .views import upload_audio

urlpatterns = [
    path('uploaded/', upload_audio.as_view(), name='Audio uploaded')
]