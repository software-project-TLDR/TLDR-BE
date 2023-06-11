from rest_framework import serializers
from .models import Transcription

class TranscriptionSerializers(serializers.ModelSerializer):
    class Meta:
        model = Transcription
        feilds = ['id', 'title', 'transcription', 'summarization']