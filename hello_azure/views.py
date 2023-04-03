import os
import pickle

import numpy as np
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django import forms

from .utils import LBP_feature, RawFeatureSVMModel, HarrisFeatureSVMModel, HogFeatureSVMModel, HistogramFeatureSVMModel, Raw_feature, Harris_feature, Hog_feature, Histogram_feature, Shitomasi_feature

from PIL import Image, ImageOps


def index(request):
    print('Request for index page received')
    return render(request, 'hello_azure/index.html')

@csrf_exempt
def hello(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        
        if name is None or name == '':
            print("Request for hello page received with no name or blank name -- redirecting")
            return redirect('index')
        else:
            print("Request for hello page received with name=%s" % name)
            context = {'name': name }
            return render(request, 'hello_azure/hello.html', context)
    else:
        return redirect('index')

@csrf_exempt
def predict(request):
    print(request.FILES)
    print(request.POST)
    print(request.encoding)
    print(request.content_type)
    print(request.content_params)
    print(request.headers)

    if request.method == 'POST':
        fileData = request.FILES['data']
        algorithm = request.POST['algorithm']

        image = Image.open(fileData)
        image = ImageOps.grayscale(image)
        image = image.resize((28, 28))

        features = np.array(image)
        constrast = lambda x: 255 - x
        features = np.array([constrast(i) for i in features])

        model = ''

        if algorithm == 'Raw SVM':
            model = RawFeatureSVMModel
            features = Raw_feature(features)
        elif algorithm == 'Harris SVM':
            model = HarrisFeatureSVMModel
            features = Harris_feature(features)
        elif algorithm == 'Histogram SVM':
            model = HistogramFeatureSVMModel
            features = Histogram_feature(features)
        elif algorithm == 'Hog SVM':
            model = HogFeatureSVMModel
            features = Hog_feature(features)
        # elif algorithm == 'Shitomasi SVM':
        #     model = ShitomasiFeatureSVMModel
        #     features = Shitomasi_feature(features)
        # elif algorithm == 'LBP SVM':
        #     model = LBPFeatureSVMModel
        #     features = LBP_feature(features)
        else: 
            model = RawFeatureSVMModel
            features = Raw_feature(features)

        result = model.predict([features])[0]

        return JsonResponse({'result': int(result), 'algorithm': algorithm})
    else:
        return JsonResponse({'error': 'Request method is wrong'})
