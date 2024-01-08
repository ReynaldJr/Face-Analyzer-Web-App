from django import forms
from faceapp.models import FaceAnalyzer

class FaceAnalyzerForm(forms.ModelForm):
    
    class Meta:
        model = FaceAnalyzer
        fields = ['image']
        