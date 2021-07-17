import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session


def index(request) :
    if request.method :

        #file = request.FILES("imageFile")
        #print(request.FILES('imageFile'))
        #fs = FileSystemStorage()
        #filePathName = fs.save(file.name, file)
        #filePathName = fs.url(filePathName)
        #testimage = '.'+filePathName

        #file_name = default_storage.save(file.name, file)
        #file_url = default_storage.path(file_name)
        #image = load_img(testimage, target_size=(224,224))
        image = load_img('media/chat.jpg', target_size=(224, 224))
        numpy_array = img_to_array(image)
        image_batch = np.expand_dims(numpy_array, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())

        with settings.GRAPH1.as_default():
            set_session(settings.SESS)
            predections = settings.IMAGE_MODEL.predict(processed_image)

        label = decode_predictions(predections, top=3)
        return render(request, "index.html", {"predections": label})
    else:
        return render(request, "index.html")
    return render(request, "index.html")