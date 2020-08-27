from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import AudioFileForm
from .classifier_webinterface import analyze_file

app_name = "Signalfelsupptäckaren"
# Create your views here.
def index(request):
	form = AudioFileForm()
	context = {
		"app_name": app_name,
		"form": form
	}
	return render(request, "simple_classifier/index.html", context)

def analysis(request):
	if request.method == 'POST':
		form = AudioFileForm(request.POST, request.FILES)
		if form.is_valid():
			analyses = analyze_file(request.FILES['audiofile'])
			return render(request, "simple_classifier/analysis.html", {
				"app_name": app_name,
				"filename": request.FILES['audiofile'].name,
				"analyses": analyses
			})

	return redirect('/')


# Because for some reason this is not included by default
# https://stackoverflow.com/a/8000091
from django.template.defaulttags import register

@register.filter
def get_item(dictionary, key):
	return dictionary.get(key)