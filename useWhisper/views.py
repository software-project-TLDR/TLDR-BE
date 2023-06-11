from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tempfile
from django.http import HttpResponse
from pydub import AudioSegment
import os
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie, csrf_protect, csrf_exempt
from django.utils.decorators import method_decorator

from django.http import Http404, JsonResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import TranscriptionSerializers
from .models import Transcription
import whisper
import time
from django.conf import settings
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TranscriptionListView(APIView):
    def post(self, request):
        serializer = TranscriptionSerializers(data = request.data)
        
        
class upload_audio(APIView):
    # @csrf_exempt
    def post(self, request):
        if request.FILES.get('audio'):
            audio_file = request.FILES['audio']
            file_extension = os.path.splitext(audio_file.name)[1]

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(audio_file.read())

            # MP3로 변환
            audio = AudioSegment.from_file(temp_file.name)
            mp3_file = audio.export(format='mp3')

            # 파일 저장
            new_file_name = f'{os.path.splitext(audio_file.name)[0]}.mp3'  # 새로운 파일 이름 생성
            saved_file_path = default_storage.save(f'{new_file_name}', ContentFile(mp3_file.read()))

            # 임시 파일 삭제
            os.remove(temp_file.name)
            
            #1초 sleep --> 혹시 파일이 저장되는데 시간이 걸릴까봐
            #time.sleep(1)
            
            #whisper 사용
            model = whisper.load_model("base")
            audio = whisper.load_audio(settings.MEDIA_ROOT + '/' + new_file_name)
            
            # pad_or_trim()을 안 써야만 30초 이상 transcribe 가능
            #audio = whisper.pad_or_trim(audio)
            
            #아래 주석된 코드들은 pad_or_trim()을 써야만 에러가 안 남.
            # # make log-Mel spectrogram and move to the same device as the model
            # mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # # detect the spoken language
            # _, probs = model.detect_language(mel)
            # print(f"Detected language: {max(probs, key=probs.get)}")

            # # decode the audio
            # options = whisper.DecodingOptions()
            # result = whisper.decode(model, mel, options, fp16=False)
            

            result_all = model.transcribe(audio, fp16=False)
            result_text = {"transcription" : result_all['text']}
            #print("Transcription : " + result_text['text'])


            #kobart 불러오기
            tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
            kobart_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
            #토크나이저를 사용해 모델이 인식할 수 있는 토큰 형태로 바꿈.
            #인코딩을 하면 숫자들의 배열로 토큰화 됨
            #참조 : https://www.dinolabs.ai/395
            input_ids = tokenizer.encode(result_text['transcription'])
            
            #모델에 넣기 전 문장의 시작과 끝을 나타내는 토큰 추가
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            input_ids = torch.tensor([input_ids])
            
            #요약문 토큰 만들기
            summary_text_ids = kobart_model.generate(
                input_ids = input_ids,
                bos_token_id = kobart_model.config.bos_token_id,
                eos_token_id = kobart_model.config.eos_token_id,
                length_penalty = 1.0, #길이에 대한 패널티값. 1보다 작은 경우 더 짧은 문장을 생성하도록 유도하며, 1보다 클 경우 길이가 더 긴 문장을 유도
                max_length = 128,
                min_length = 32,
                num_beams = 4,  #문장 생성시 다음 단어를 탐색하는 영역의 개수
            )
            
            result_text['summarization'] = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
            
            #print("Summarization : " + result_text['summarization'])
            print(result_text)
            
            save_in_DB = Transcription(
                title = new_file_name,
                transcription = result_text['transcription'],
                summarization = result_text['summarization']
            )
            save_in_DB.save()
            
            print(Transcription.objects.filter(pk=save_in_DB.id).values())
            
            return JsonResponse(result_text, safe=False, json_dumps_params={'ensure_ascii': False}, status=200)
        

        return HttpResponse(status=400)

    def get(self, request):
        return HttpResponse("Success",status=200)
    
# 만약 GET을 통해 쿼리 불러올 시 처리할 명령어    
# class result_page(APIView):
#     def get(self, request):
#         q = Transcription.objects.filter(title=request.GET.get('id'))
#         return JsonResponse(q, safe=False, json_dumps_params={'ensure_ascii': False}, status=200)