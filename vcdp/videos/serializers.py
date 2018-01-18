from rest_framework import serializers
from django.core.validators import validate_comma_separated_integer_list
from django.contrib.auth.models import User
from videos.models import Video
from videos.models import Dataset


class VideoSerializer(serializers.ModelSerializer):

    class Meta:
        model = Video
        fields = ('id', 'vp_video_id', 'title', 'description', 'comments', 'label', 'dataset', 'subset_type')


class DatasetSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    videos = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = ('id', 'title', 'videos', 'owner', 'is_prepared')


class DatasetSerializerVerbose(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    videos = VideoSerializer(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = ('id', 'title', 'videos', 'owner', 'is_prepared')


class UserSerializer(serializers.ModelSerializer):
    datasets = serializers.PrimaryKeyRelatedField(many=True, queryset=Dataset.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'password', 'datasets')
        extra_kwargs = {'password': {'write_only': True,},}


class PrepareDatasetSerializer(serializers.Serializer):
    train_size = serializers.IntegerField()
    test_size = serializers.IntegerField()


class TrainSerializer(serializers.Serializer):
    clf_type = serializers.IntegerField() # TODO upper lower limit


class AnalysisSerializer(serializers.Serializer):
    accuracy = serializers.CharField(max_length=10)
    classification_report = serializers.CharField(max_length=256)
    confusion_matrix = serializers.CharField(max_length=256, validators=[validate_comma_separated_integer_list])


class AnalysesSerializer(serializers.Serializer):
    MultinominalNb = AnalysisSerializer()
    SGDClassifier = AnalysisSerializer()


class PredictSerializer(serializers.Serializer):
    clf_type = serializers.IntegerField() # TODO upper lower limit
    vp_video_ids = serializers.ListField(child=serializers.CharField())
