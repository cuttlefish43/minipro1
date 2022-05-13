from django.db import models

# Create your models here.
class SummarizationDataset(models.Model):
    inpText=models.TextField()
    inpTextLength=models.IntegerField()
    algo1ExtTxt=models.TextField()
    algo1ExtLength=models.IntegerField()
    algo2ExtTxt=models.TextField()
    algo2ExtLength=models.IntegerField()
    algo3AbsTxt=models.TextField()
    algo3AbsLength=models.IntegerField()
    UserChoice=models.IntegerField()
    SentimentScore=models.FloatField()
    