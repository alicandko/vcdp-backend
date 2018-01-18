import random
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from django.conf import settings


'''DATA MODELS'''

class Dataset(models.Model):
    title = models.CharField(max_length=256, default='')
    owner = models.ForeignKey('auth.User', related_name='datasets', on_delete=models.CASCADE)
    is_prepared = models.BooleanField(default=False)


class Video(models.Model):
    vp_video_id = models.CharField(max_length=20, default='')
    title = models.CharField(max_length=256, default='')
    description = models.CharField(max_length=256, default='', blank=True)
    comments = models.TextField(default='', blank=True)
    label = models.CharField(max_length=20, default='')
    dataset = models.ForeignKey(Dataset, related_name='videos', on_delete=models.CASCADE)
    subset_type = models.CharField(max_length=10, default='', blank=True)

# This code is triggered whenever a new user has been created and saved to the database
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)



'''DOMAIN MODELS'''

class ScikitDataset():

    def __init__(self, train_videos, validate_videos, test_videos):
        self.train = {'set_type': 'train', 'data': None, 'target': None, 'target_names': None}
        self.validate = {'set_type': 'validate', 'data': None, 'target': None, 'target_names': None}
        self.test = {'set_type': 'test', 'data': None, 'target': None, 'target_names': None}
        self.__init(train_videos, validate_videos, test_videos)

    def __init(self, train_videos, validate_videos, test_videos):
        videos = train_videos + validate_videos + test_videos
        categories = list(set(map(lambda video : video['label'], videos)))
        self.train['target_names'] = categories
        self.validate['target_names'] = categories
        self.test['target_names'] = categories
        self.__fill_subset(self.train, train_videos)
        self.__fill_subset(self.validate, validate_videos)
        self.__fill_subset(self.test, test_videos)

    def __fill_subset(self, subset, videos):
        data = []
        target = []
        for video in videos:
            data.append(video['title']+ " " + video['description'])
            target.append(subset['target_names'].index(video['label']))

        subset['data'] = data
        subset['target'] = target


class Analysis():

    def __init__(self, accuracy, classification_report, confusion_matrix):
        self.accuracy = accuracy
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix


class Analyses():

    def __init__(self, multinominalnb, sgdclassifier):
        self.MultinominalNb = multinominalnb
        self.SGDClassifier = sgdclassifier
