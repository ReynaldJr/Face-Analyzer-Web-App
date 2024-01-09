from django import forms
from faceapp.models import FaceAnalyzer

class FaceAnalyzerForm(forms.ModelForm):
    
    class Meta:
        model = FaceAnalyzer
        fields = ['image']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.fields['image'].widget.attrs.update({'class':'form-control'})