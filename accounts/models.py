from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.


class CustomUser(AbstractUser):
     is_admin =models.BooleanField(default=False)
     phone=models.IntegerField(null=True)

     def __str__(self):
          return self.first_name

class History(models.Model):
     user=models.ForeignKey(CustomUser,on_delete=models.CASCADE)
     image = models.FileField(upload_to="prediction_Image")
     result = models.CharField(max_length=200)
     timestamp = models.DateTimeField(auto_now_add=True)
     model_key = models.CharField(max_length=255,null=True)
     