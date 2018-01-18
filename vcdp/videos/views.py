import logging
import sys
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework.decorators import list_route
from rest_framework.decorators import detail_route
from rest_framework import status
from rest_framework import permissions
from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import PermissionDenied
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.contrib.auth.hashers import make_password
from apiclient.discovery import build
from videos.serializers import DatasetSerializer
from videos.serializers import DatasetSerializerVerbose
from videos.serializers import VideoSerializer
from videos.serializers import UserSerializer
from videos.serializers import PrepareDatasetSerializer
from videos.serializers import TrainSerializer
from videos.serializers import AnalysisSerializer
from videos.serializers import AnalysesSerializer
from videos.serializers import PredictSerializer
from videos.action import VideoDataExtractor
from videos.action import YoutubeVideoSearcher
from videos.action import MachineLearningManager
from videos.models import Video
from videos.models import Dataset
from videos.permissions import IsOwner
from videos.permissions import IsUser
from videos.permissions import IsCreationOrIsAuthenticated
from videos import settings



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsUser, IsCreationOrIsAuthenticated)

    def perform_create(self, serializer):
        password = make_password(self.request.data['password'])
        serializer.save(password=password)

    def list(self, request):
        if self.request.user.is_staff == True:
            serializer = UserSerializer(self.queryset, many=True)
            response = Response(serializer.data)
        else:
            response = Response({'detail': 'You do not have permission to perform this action.'},
                                status=status.HTTP_403_FORBIDDEN)
        return response

    def perform_update(self, serializer):
        password = make_password(self.request.data['password'])
        serializer.save(password=password)


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = (permissions.IsAuthenticated, IsOwner)

    machine_learning_manager = MachineLearningManager()

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    def list(self, request):
        dataset_queryset_of_user = self.queryset.filter(owner=self.request.user)
        serializer = DatasetSerializer(dataset_queryset_of_user, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        dataset = get_object_or_404(self.queryset, pk=pk)
        if dataset.owner == request.user:
            serializer = DatasetSerializerVerbose(dataset)
            response = Response(serializer.data)
        else:
            response = Response({'detail': 'You do not have permission to perform this action.'},
                                status=status.HTTP_403_FORBIDDEN)
        return response

    @detail_route()
    def videos(self, request, pk):
        logger.info("Endpoint datasets videos called")
        dataset = get_object_or_404(self.queryset, pk=pk)
        if dataset.owner == request.user:
            videos_queryset = dataset.videos.all()
            serializer = VideoSerializer(videos_queryset, many=True)
            response = Response(serializer.data)
        else:
            response = Response({'detail': 'You do not have permission to perform this action.'},
                                status=status.HTTP_403_FORBIDDEN)
        return response

    @detail_route(methods=['post'])
    def prepare_dataset(self, request, pk):
        logger.info("Endpoint datasets prepare_dataset called")
        dataset = get_object_or_404(self.queryset, pk=pk)
        if dataset.owner == request.user:
            prepare_dataset_serializer = PrepareDatasetSerializer(data=request.data)
            prepare_dataset_serializer.is_valid(raise_exception=True)
            try:
                self.machine_learning_manager.prepare_dataset(dataset,
                                                              prepare_dataset_serializer.validated_data['train_size'],
                                                              prepare_dataset_serializer.validated_data['test_size'])
                response = Response({'detail': 'dataset prepared'}, status=status.HTTP_200_OK)
            except ValueError as e:
                response = Response({'detail': e.message}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = Response({'detail': 'You do not have permission to perform this action.'},
                                status=status.HTTP_403_FORBIDDEN)

        return response

    @detail_route()
    def validate_analyse(self, request, pk):
        logger.info('Endpoint datasets validate_analyse called')
        dataset = get_object_or_404(self.queryset, pk=pk)
        try:
            if dataset.owner == request.user:
                analyses = self.machine_learning_manager.validate_analyse(dataset)
                serializer = AnalysesSerializer(analyses)
                response = Response(serializer.data, status=status.HTTP_200_OK)
            else:
                response = Response({'detail': 'You do not have permission to perform this action.'},
                                    status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            status_msg = 'dataset ' + str(pk) + ' is not prepared'
            response = Response({'detail': e.message}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

        return response

    @detail_route(methods=['post'])
    def test_analyse(self, request, pk):
        logger.info('Endpoint datasets test_analyse called')
        serializer = TrainSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset = get_object_or_404(self.queryset, pk=pk)
        try:
            if dataset.owner == request.user:
                analysis = self.machine_learning_manager.test_analyse(serializer.validated_data['clf_type'], dataset)
                serializer = AnalysisSerializer(analysis)
                response = Response(serializer.data, status=status.HTTP_200_OK)
            else:
                response = Response({'detail': 'You do not have permission to perform this action.'},
                                    status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            status_msg = 'dataset ' + str(pk) + ' is not prepared'
            response = Response({'detail': e.message}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

        return response

    @detail_route(methods=['post'])
    def predict(self, request, pk):
        logger.info('Endpoint datasets predict called')
        serializer = PredictSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset = get_object_or_404(self.queryset, pk=pk)
        try:
            if dataset.owner == request.user:
                prediction = self.machine_learning_manager.predict(serializer.validated_data['clf_type'],
                                                                   dataset, serializer.validated_data['vp_video_ids'])
                response = Response(prediction, status=status.HTTP_200_OK)
            else:
                response = Response({'detail': 'You do not have permission to perform this action.'},
                                    status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            status_msg = 'dataset ' + str(pk) + ' is not prepared'
            response = Response({'detail': e.message}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

        return response


class VideoViewSet(viewsets.ModelViewSet):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    permission_classes = (permissions.IsAuthenticated,)

    video_data_extractor = VideoDataExtractor(
                               YoutubeVideoSearcher(
                                   build(settings.YOUTUBE_API_SERVICE_NAME,
                                       settings.YOUTUBE_API_VERSION,
                                       developerKey=settings.DEVELOPER_KEY)))

    machine_learning_manager = MachineLearningManager()

    def list(self, request):
        dataset_query_set = Dataset.objects.all()
        dataset_queryset_of_user = dataset_query_set.filter(owner=self.request.user)
        videos = []
        for dataset in dataset_queryset_of_user:
            videos_of_dataset = dataset.videos.all()
            serializer = VideoSerializer(videos_of_dataset, many=True)
            videos = videos + serializer.data
        return Response(videos)

    def retrieve(self, request, pk=None):
        video = get_object_or_404(self.queryset, pk=pk)
        if video.dataset.owner == request.user:
            serializer = VideoSerializer(video)
            response = Response(serializer.data)
        else:
            response = Response({'detail': 'You do not have permission to perform this action.'},
                                status=status.HTTP_403_FORBIDDEN)
        return response

    def perform_create(self, serializer):
        isAllowed = True
        if isinstance(self.request.data, list):
            for video in self.request.data:
                dataset_id = video['dataset']
                dataset_query_set = Dataset.objects.all()
                dataset = get_object_or_404(dataset_query_set, pk=dataset_id)
                if self.request.user != dataset.owner:
                    isAllowed = False
                    raise PermissionDenied()
                self.machine_learning_manager.set_dataset_is_prepared(dataset, False)
        else :
            dataset_id = self.request.data['dataset']
            dataset_query_set = Dataset.objects.all()
            dataset = get_object_or_404(dataset_query_set, pk=dataset_id)
            if self.request.user != dataset.owner:
                isAllowed = False
                raise PermissionDenied()
            self.machine_learning_manager.set_dataset_is_prepared(dataset, False)

        if isAllowed:
            serializer.save()

    def update(self, request, pk=None):
        video = get_object_or_404(self.queryset, pk=pk)
        dataset = video.dataset
        if dataset.owner == request.user:
            serializer = VideoSerializer(video, data=request.data)
            serializer.is_valid(raise_exception=True)
            dataset_query_set = Dataset.objects.all()
            new_dataset = get_object_or_404(dataset_query_set, pk=request.data['dataset'])
            if new_dataset.owner == request.user:
                serializer.save()
                response = Response(serializer.data)
            else:
                raise PermissionDenied()
        else:
            raise PermissionDenied()

        return response

    def update(self, request, pk=None):
        video = get_object_or_404(self.queryset, pk=pk)
        dataset = video.dataset
        if dataset.owner == request.user:
            serializer = VideoSerializer(video, data=request.data)
            serializer.is_valid(raise_exception=True)
            dataset_query_set = Dataset.objects.all()
            new_dataset = get_object_or_404(dataset_query_set, pk=request.data['dataset'])
            if new_dataset.owner == request.user:
                serializer.save()
                response = Response(serializer.data)
            else:
                raise PermissionDenied()
        else:
            raise PermissionDenied()

        return response

    def destroy(self, request, pk=None):
        video = get_object_or_404(self.queryset, pk=pk)
        dataset = video.dataset
        if dataset.owner == request.user:
            video.delete()
            response = Response(status=status.HTTP_204_NO_CONTENT)
        else:
            raise PermissionDenied()

        return response

    @list_route()
    def search(self, request):
        logger.info("Endpoint videos search triggered")
        videos = self.video_data_extractor.extract(request.query_params.get('q'))
        serializer = VideoSerializer(videos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def get_serializer(self, *args, **kwargs):
        if "data" in kwargs:
            data = kwargs["data"]
            if isinstance(data, list):
                kwargs["many"] = True
        return super(VideoViewSet, self).get_serializer(*args, **kwargs)
