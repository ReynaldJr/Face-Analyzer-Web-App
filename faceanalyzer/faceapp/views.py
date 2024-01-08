from django.shortcuts import render
from django.http import HttpResponse
from faceapp.forms import FaceAnalyzerForm
from faceapp.ml_model import face_recognizer_pipeline_model
from django.conf import settings
from faceapp.models import FaceAnalyzer
import os

# Create your views here.

def index(request):
    form = FaceAnalyzerForm()
    
    if request.method == 'POST':
        form = FaceAnalyzerForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            
            # Extracting the image from database
            primary_key = save.pk
            imgobj = FaceAnalyzer.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)
            results = face_recognizer_pipeline_model(filepath)
            print(results)
            
            return render(request,'index.html',{'form':form,'upload':True, 'results':results})
    
    return render(request,'index.html',{'form': form, 'upload':False})