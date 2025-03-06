from django.shortcuts import render,redirect
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login
from .models import *
from .forms import *
import os
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from PIL import Image
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.utils.timezone import now
from datetime import timedelta
from django.core.paginator import Paginator

from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.shortcuts import render,redirect,get_object_or_404
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login
from .models import *
from .forms import *

class LoginView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        log_form=LogForm(data=request.POST)
        if log_form.is_valid():  
            us=log_form.cleaned_data.get('username')
            ps=log_form.cleaned_data.get('password')
            user=authenticate(request,username=us,password=ps)
            if user: 
                login(request,user)
                return redirect('main')
            else:
                return render(request,'login.html',{"form":log_form})
        else:
            return render(request,'login.html',{"form":log_form}) 
        

class RegView(CreateView):
     form_class=UserForm
     template_name="reg.html"
     model=CustomUser
     success_url=reverse_lazy("login")  


class MainPage(TemplateView):
    template_name = 'home.html'
    

class Prediction(TemplateView):
    template_name = 'prediction.html'
    




   
# class CheckUpView(View):
#     def get(self, request):
#         return render(request, "checkup.html")
#     def post(self, request):
#         image = request.FILES.get('image')     
#         if not image:
#             return render(request, "checkup.html", {'error': 'Please upload an image.'})
#         cat=classify_image(model_path,image)
#         return render(request,"checkup.html",{'response': cat})
    


from django.contrib.auth import logout as auth_logout


class HistoryView(TemplateView):
    template_name = 'history.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        user = self.request.user
        query_params = self.request.GET

        # Get initial queryset
        history_qs = History.objects.filter(user=user).order_by('-timestamp')

        # Filtering by date range
        date_range = query_params.get('date_range', 'all')
        if date_range == "last_7_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=7))
        elif date_range == "last_30_days":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=30))
        elif date_range == "last_3_months":
            history_qs = history_qs.filter(timestamp__gte=now() - timedelta(days=90))

        # Filtering by model type
        model_type = query_params.get('model_type', '')
        if model_type and model_type != "All Models":
            history_qs = history_qs.filter(model_key=model_type)

        # Search by user input
        search_query = query_params.get('search', '')
        if search_query:
            history_qs = history_qs.filter(result__icontains=search_query)

        # Pagination
        paginator = Paginator(history_qs, 10)  # 10 items per page
        page_number = query_params.get('page', 1)
        history_page = paginator.get_page(page_number)

        context['history'] = history_page
        context['page_obj'] = history_page
        return context
    

def custom_logout(request):
    auth_logout(request)
    return redirect('login')

models = {
    "Breast_cancer": {
        "model": load_model("D:\Projects\Can Care\cancare\models\BreastCancer_model .h5"),
        "class_labels": ['malignant', 'benign'],
        "input_shape": (1,150, 150,1)
    },
    "Brain_tumor": {
        "model": load_model("D:\Projects\Can Care\cancare\models\BrainTumor_model .h5"),
        "class_labels": ['glioma', 'meningioma', 'notumor', 'pituitary'],
        "input_shape": (1,150, 150,1)
    },
    "Kidney_cancer": {
        "model": load_model("D:\Projects\Can Care\cancare\models\KidneyCancer_model .h5"),
        "class_labels": ['Normal', 'Tumor'],
        "input_shape": (1,150, 150,1)
    },
    "Lung_cancer": {
        "model": load_model("D:\Projects\Can Care\cancare\models\LungCancer_model1.h5"),
        "class_labels": ['adenocarcinoma', 'benign', 'squamous_carcinoma'],
        "input_shape": (1,150, 150,3)
    },
    "Skin_cancer": {
        "model": load_model("D:\Projects\Can Care\cancare\models\MelanomaCancer_model1.h5"),
        "class_labels": ['benign', 'malignant'],
        "input_shape": (1,150,150,3)
    }
}
from skimage.transform import resize


# Function to normalize the image for model prediction
def normalize_image(image, model_key, models):
    """Normalize image values to [0, 1] range."""
    model_info = models.get(model_key)
    
    if model_info is None:
        raise ValueError(f"Model info for key {model_key} not found.")
    
    # Resize the image to match the input shape required by the model
    img_resized = resize(image, model_info["input_shape"][:2])
    
    # Normalize the image by dividing by 255 to bring values to [0, 1]
    img_array = img_resized / 255.0
    
    return img_array


# Preprocess the image for prediction
def preprocess_image(imagefile, model_key):
    """Preprocess the image for model prediction."""
    try:
        img_array = normalize_image(imagefile, model_key)
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        raise Exception(f"Error during image preprocessing: {e}")

@login_required
def predict(request):
    """Handle prediction requests and return results."""
    if request.method == "POST":
        try:
            model_key = request.POST.get("model_key")
            imagefile = request.FILES.get("imagefile")

            if not model_key or not imagefile:
                return render(request, "prediction.html", {"error": "Model key and image file are required."})

            model = models[model_key]["model"]
            class_labels = models[model_key]["class_labels"]

            # Preprocess the image
            img_array = preprocess_image(imagefile, model_key)

            # Make prediction
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100
            prediction_result = class_labels[class_index]

            # Save the image and prediction in history
            prediction = History.objects.create(
                user=request.user,
                model_key=model_key,
                result=prediction_result,
                image=imagefile,
            )

            return render(request, "prediction.html", {
                "prediction": prediction_result,
                "confidence": f"{confidence:.2f}%",
                "image_url": prediction.image.url,
                "user": request.user,
                "model_key": model_key,
            })

        except Exception as e:
            return render(request, "prediction.html", {"error": str(e)})

    return render(request, "prediction.html")

def remove_History(req,pk):
    try:
        sub= get_object_or_404(History,id=pk)
        sub.delete()
        # sub.save()
        return redirect('history')
    except Exception as e:
        return HttpResponse(f"An error occurred: {str(e)}", status=500)
    
import re
import groq
    
client = groq.Client(api_key="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")


def get_groq_response(user_input):
    """
    Communicate with the GROQ chatbot to get a response based on user input.
    """
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. You reply with very short answers ."
    }

    chat_history = [system_prompt]

    # Append user input to the chat history
    chat_history.append({"role": "user", "content": user_input})

    # Get response from GROQ API
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=chat_history,
        max_tokens=100,
        temperature=1.2
    )

    response = chat_completion.choices[0].message.content
    print(response)
    # Format response (convert **bold** to <b>bold</b>)
    response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)

    return response

import json
from django.http import JsonResponse

class ChatbotView(View):
    def get(self, request):
        return render(request, "chatbot.html")
    def post(self, request): 
        try:
            body = json.loads(request.body)
            user_input = body.get('userInput')
        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Invalid JSON format."})
    
        if not user_input:  # If user_input is None or empty
            print("no")
            return JsonResponse({"error": "No user input provided."})  
        
        print("User Input:", user_input)
        
        static_responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! How can I help you?",
            "how are you": "I'm just a chatbot, but I'm doing great! How about you?",
            "bye": "Goodbye! Take care.",
            "whats up": "Not much, just here to help you with  queries. How can I help you today?",
        }

        lower_input = user_input.lower().strip()
        if lower_input in static_responses:
            print(static_responses[lower_input])
            return JsonResponse({'response': static_responses[lower_input]})
        
        try:
            print("Processing via GROQ")
            data = get_groq_response(user_input)
            print(data)
            treatment_list = data.split('\n')
            return JsonResponse({'response': treatment_list})
        except Exception as e:
            return JsonResponse({"error": f"Failed to get GROQ response: {str(e)}"})