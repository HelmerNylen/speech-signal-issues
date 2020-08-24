from django import forms

class AudioFileForm(forms.Form):
	audiofile = forms.FileField(label="Ljudfil")