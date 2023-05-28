from .views import GetData
from django.urls import path,include,re_path

urlpatterns = [  
    path('predict', GetData.as_view()),  
]  