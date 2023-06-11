from django.db import models

# Create your models here.
class Transcription(models.Model):
    title = models.CharField(max_length=50)
    transcription = models.TextField(blank=True)
    summarization = models.TextField(blank=True)
    our_summarization = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title