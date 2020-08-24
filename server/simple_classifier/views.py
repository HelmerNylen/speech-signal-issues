from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import AudioFileForm
from .classifier_webinterface import analyze_file

# Create your views here.
def index(request):
	if request.method == 'POST':
		form = AudioFileForm(request.POST, request.FILES)
		if form.is_valid():
			predictions, scores = analyze_file(request.FILES['audiofile'])
			return render(request, "simple_classifier/analysis.html", {
				"filename": request.FILES['audiofile'].name,
				"predictions": predictions,
				"scores": scores
			})
	else:
		form = AudioFileForm()
		
	context = {
		"app_name": "Signalfelsupptäckaren",
		"form": form
	}
	return render(request, "simple_classifier/index.html", context)


# Because for some reason this is not included by default
# https://stackoverflow.com/a/8000091
from django.template.defaulttags import register

@register.filter
def get_item(dictionary, key):
	return dictionary.get(key)