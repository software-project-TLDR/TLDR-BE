from django.urls import path
from .views import upload_audio
#from .views import result_page
urlpatterns = [
    path('uploaded/', upload_audio.as_view(), name='Audio uploaded'),
    #get response 들어올 시 경로 설정
    #path('resultpage/', result_page.as_view(), name='result_get')
]