from django.test import TestCase
from rest_framework.test import APIRequestFactory
from rest_framework.test import force_authenticate
from mock import MagicMock
from apiclient.discovery import build
from videos.models import Video
from videos.models import Dataset
from videos.models import Analyses
from videos.models import Analysis
from django.contrib.auth.models import User
from videos.serializers import VideoSerializer
from videos.serializers import DatasetSerializer
from videos.serializers import UserSerializer
from videos.action import VideoDataExtractor
from videos.action import YoutubeVideoSearcher
from videos.action import MachineLearningManager
from videos.views import UserViewSet
from videos.views import DatasetViewSet
from videos.views import VideoViewSet
from videos import settings


class VideoTestCase(TestCase):

    def setUp(self):
        user1 = User.objects.create(id=1, username="username", password="password")
        user1.save()

        user2 = User.objects.create(id=2, username="anotherUsername", password="password")
        user2.save()

        staff = User.objects.create(id=3, username="staff", password="password", is_staff=True)
        staff.save()

        dataset1 = Dataset.objects.create(id=1, title="datasetTitle1", owner=user1 ,is_prepared=False)
        dataset1.save()

        dataset2 = Dataset.objects.create(id=2, title="datasetTitle2", owner=user1 ,is_prepared=False)
        dataset2.save()

        dataset3 = Dataset.objects.create(id=3, title="datasetTitle3", owner=user2 ,is_prepared=False)
        dataset3.save()

        label_suffix = 1
        for i in range(1, 101):
            if label_suffix > 3:
                label_suffix = 1
            video = Video.objects.create(id=i, vp_video_id="vpVideoId" + str(i), title="videoTitle" + str(i),
                                         description="description" + str(i), comments="comments" + str(i),
                                         label="label" + str(label_suffix), dataset=dataset1, subset_type="")
            video.save()
            label_suffix = label_suffix + 1

        video = Video.objects.create(id=101, vp_video_id="vpVideoId", title="videoTitle",
                                     description="description", comments="comments",
                                     label="label", dataset=dataset2, subset_type="")
        video.save()

        video = Video.objects.create(id=102, vp_video_id="vpVideoId", title="videoTitle",
                                             description="description", comments="comments",
                                             label="label", dataset=dataset3, subset_type="")
        video.save()

    def tearDown(self):
        Video.objects.all().delete()
        Dataset.objects.all().delete()
        User.objects.all().delete()


    '''Models'''
    def test_models(self):
        user = User.objects.get(pk=1)
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "username")
        self.assertEqual(user.password, "password")

        dataset = Dataset.objects.get(pk=1)
        self.assertEqual(dataset.title, "datasetTitle1")
        self.assertEqual(dataset.owner, user)
        self.assertEqual(dataset.is_prepared, False)

        video = Video.objects.get(pk=1)
        self.assertEqual(video.id, 1)
        self.assertEqual(video.vp_video_id, "vpVideoId1")
        self.assertEqual(video.title, "videoTitle1")
        self.assertEqual(video.description, "description1")
        self.assertEqual(video.comments, "comments1")
        self.assertEqual(video.label, "label1")
        self.assertEqual(video.dataset, dataset)
        self.assertEqual(video.subset_type, "")

        video = Video.objects.get(pk=2)
        self.assertEqual(video.id, 2)
        self.assertEqual(video.vp_video_id, "vpVideoId2")
        self.assertEqual(video.title, "videoTitle2")
        self.assertEqual(video.description, "description2")
        self.assertEqual(video.comments, "comments2")
        self.assertEqual(video.label, "label2")
        self.assertEqual(video.dataset, dataset)
        self.assertEqual(video.subset_type, "")


    '''Serializers'''
    def test_serializers(self):
        serializer = VideoSerializer(Video.objects.get(pk=1))
        self.assertEqual(serializer.data.get('id'), 1)
        self.assertEqual(serializer.data.get('vp_video_id'), "vpVideoId1")
        self.assertEqual(serializer.data.get('title'), "videoTitle1")
        self.assertEqual(serializer.data.get('description'), "description1")
        self.assertEqual(serializer.data.get('comments'), "comments1")
        self.assertEqual(serializer.data.get('label'), "label1")
        self.assertEqual(serializer.data.get('dataset'), 1)
        self.assertEqual(serializer.data.get('subset_type'), "")

        serializer = DatasetSerializer(Dataset.objects.get(pk=1))
        self.assertEqual(serializer.data.get('id'), 1)
        self.assertEqual(serializer.data.get('title'), "datasetTitle1")
        self.assertEqual(serializer.data.get('is_prepared'), False)

        serializer = UserSerializer(User.objects.get(pk=1))
        self.assertEqual(serializer.data.get('id'), 1)
        self.assertEqual(serializer.data.get('username'), "username")


    '''VideoDataExtractor'''
    def test_extract_returns_videos(self):
        youtube_video_searcher = YoutubeVideoSearcher(
                                     build(settings.YOUTUBE_API_SERVICE_NAME,
                                           settings.YOUTUBE_API_VERSION,
                                           developerKey=settings.DEVELOPER_KEY))
        videos = []
        video = Video(
            vp_video_id="vpVideoId",
            title="title",
            description="description",
            comments="",
            label=""
        )
        videos.append(video)
        videos.append(video)

        youtube_video_searcher.search = MagicMock(return_value=videos)

        video_data_extractor = VideoDataExtractor(youtube_video_searcher)
        actual = video_data_extractor.extract("hello")
        self.assertEqual(actual[0], video)
        self.assertEqual(actual[1], video)


    '''MachineLearningManager'''
    def test_prepare_dataset_assigns_subset_type(self):
        machine_learning_manager = MachineLearningManager()
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager.prepare_dataset(dataset, 60, 20)
        self.assertEqual(dataset.is_prepared, True)
        train_count = 0
        validate_count = 0
        test_count = 0

        for i in range(1, 101):
            video = Video.objects.get(pk=i)
            if video.subset_type == "train":
                train_count = train_count + 1
            if video.subset_type == "validate":
                validate_count = validate_count + 1
            if video.subset_type == "test":
                test_count = test_count + 1

        self.assertEqual(train_count, 60)
        self.assertEqual(validate_count, 20)
        self.assertEqual(test_count, 20)

    def test_set_dataset_is_prepared_sets_is_prepared(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        machine_learning_manager.set_dataset_is_prepared(dataset, False)
        self.assertEqual(dataset.is_prepared, False)
        machine_learning_manager.set_dataset_is_prepared(dataset, True)
        self.assertEqual(dataset.is_prepared, True)

    def test_validate_analyse_returns_analyses(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        machine_learning_manager.prepare_dataset(dataset, 60, 20)
        self.assertIsInstance(machine_learning_manager.validate_analyse(dataset), Analyses)

    def test_validate_analyse_raises_exception_if_not_prepared(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        with self.assertRaises(Exception):
            machine_learning_manager.validate_analyse(dataset)

    def test_test_analyse_returns_analysis(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        machine_learning_manager.prepare_dataset(dataset, 60, 20)
        self.assertIsInstance(machine_learning_manager.test_analyse(0, dataset), Analysis)
        self.assertIsInstance(machine_learning_manager.test_analyse(1, dataset), Analysis)

    def test_test_analyse_raises_exception_if_not_prepared(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        with self.assertRaises(Exception):
            machine_learning_manager.test_analyse(dataset)

    def test_predict_returns_predictions(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        machine_learning_manager.prepare_dataset(dataset, 60, 20)
        self.assertIsInstance(machine_learning_manager.predict(1, dataset, ['KDxJlW6cxRk', 'NYhxaZXXwsg']), list)

    def test_predict_raises_exception_if_not_prepared(self):
        dataset = Dataset.objects.get(pk=1)
        machine_learning_manager = MachineLearningManager()
        with self.assertRaises(Exception):
            machine_learning_manager.predict(1, dataset, ['KDxJlW6cxRk', 'NYhxaZXXwsg'])


    '''UserViewSet'''
    def test_get_users_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/users/')
        staff = User.objects.get(pk=3)
        force_authenticate(request, user=staff)
        view = UserViewSet.as_view({'get': 'list'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '[{"id":1,"username":"username","datasets":[1,2]},' +
                                            '{"id":2,"username":"anotherUsername","datasets":[3]},' +
                                            '{"id":3,"username":"staff","datasets":[]}]')

    def test_get_users_returns_forbidden_if_not_admin(self):
        factory = APIRequestFactory()
        request = factory.get('/users/')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = UserViewSet.as_view({'get': 'list'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_get_user_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/users/1')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = UserViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk='1')
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":1,"username":"username","datasets":[1,2]}')

    def test_get_user_returns_forbidden_if_not_user(self):
        factory = APIRequestFactory()
        request = factory.get('/users/1')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = UserViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk='1')
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_create_user_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/users/', {'username': "username4", 'password': "password4"})
        view = UserViewSet.as_view({'post': 'create'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.content, '{"id":4,"username":"username4","datasets":[]}')

    def test_update_user_returns_success(self):
        factory = APIRequestFactory()
        request = factory.put('/users/1', {'username': "updatedUsername", 'password': "password"})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = UserViewSet.as_view({'put': 'update'})
        response = view(request, pk=1)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":1,"username":"updatedUsername","datasets":[1,2]}')

    def test_update_user_returns_forbidden_if_not_user(self):
        factory = APIRequestFactory()
        request = factory.put('/users/1', {'username': "updatedUsername", 'password': "password"})
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = UserViewSet.as_view({'put': 'update'})
        response = view(request, pk=1)
        response.render()
        self.assertEqual(response.status_code, 403)


    '''DatasetViewSet'''
    def test_get_datasets_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/datasets/')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '[{"id":1,"title":"datasetTitle1","videos":[1,2,3,4,5,6,7,8,9,10,11,' +
                                           '12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,' +
                                           '34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,' +
                                           '56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,' +
                                           '78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,' +
                                           '100],"owner":"username","is_prepared":false},{"id":2,' +
                                           '"title":"datasetTitle2","videos":[101],"owner":"username",' +
                                           '"is_prepared":false}]')

    def test_get_dataset_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/datasets/2')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=2)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":2,"title":"datasetTitle2","videos":[{"id":101,' +
                                           '"vp_video_id":"vpVideoId","title":"videoTitle",' +
                                           '"description":"description","comments":"comments","label":"label",' +
                                           '"dataset":2,"subset_type":""}],"owner":"username","is_prepared":false}')

    def test_get_dataset_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.get('/datasets/2')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=2)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_create_dataset_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/', {'title': "datasetTitle4", 'owner': 1, 'is_prepared': False})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.content, '{"id":4,"title":"datasetTitle4","videos":[],"owner":"username",' +
                                           '"is_prepared":false}')

    def test_update_dataset_returns_success(self):
        factory = APIRequestFactory()
        request = factory.put('/datasets/2', {'title': "updatedDatasetTitle", 'owner': 1, 'is_prepared': False})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'put': 'update'})
        response = view(request, pk=2)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":2,"title":"updatedDatasetTitle","videos":[101],' +
                                           '"owner":"username","is_prepared":false}')

    def test_update_dataset_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.put('/datasets/2', {'title': "updatedDatasetTitle", 'is_prepared': False})
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'put': 'update'})
        response = view(request, pk=2)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_delete_dataset_returns_success(self):
        factory = APIRequestFactory()
        request = factory.delete('/datasets/2')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=2)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(len(Dataset.objects.all()), 2)

    def test_delete_dataset_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.delete('/datasets/2')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=2)
        self.assertEqual(response.status_code, 403)

    def test_prepare_dataset_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 200)

    def test_prepare_dataset_returns_bad_request_if_data_invalid(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train': 60, 'test': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 400)

    def test_prepare_dataset_returns_bad_request_if_sizes_invalid(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 100, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 400)

    def test_validate_analyse_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)

        request = factory.get('/datasets/1/validate_analyse')
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'get': 'validate_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 200)

    def test_validate_analyse_returns_method_not_allowed_if_not_prepared(self):
        factory = APIRequestFactory()
        request = factory.get('/datasets/1/validate_analyse')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'get': 'validate_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 405)

    def test_test_analyse_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)

        request = factory.post('/datasets/1/test_analyse', {'clf_type': 0})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'test_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 200)

        request = factory.post('/datasets/1/test_analyse', {'clf_type': 1})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'test_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 200)

    def test_test_analyse_returns_method_not_allowed_if_not_prepared(self):
        factory = APIRequestFactory()
        user = User.objects.get(pk=1)
        request = factory.post('/datasets/1/test_analyse', {'clf_type': 0})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'test_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 405)

    def test_test_analyse_returns_bad_request_if_data_invalid(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)

        request = factory.post('/datasets/1/test_analyse', {'clf': 0})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'test_analyse'})
        response = view(request, pk=1)
        self.assertEqual(response.status_code, 400)

    def test_predict_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)

        request = factory.post('/datasets/1/predict', {'vp_video_ids': ['KDxJlW6cxRk', 'NYhxaZXXwsg'], 'clf_type': 1})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'predict'})
        response = view(request, pk=1)
        response.render()
        self.assertEqual(response.status_code, 200)

    def test_predict_returns_method_not_allowed_if_not_prepared(self):
        factory = APIRequestFactory()
        user = User.objects.get(pk=1)
        request = factory.post('/datasets/1/predict', {'vp_video_ids': ['KDxJlW6cxRk', 'NYhxaZXXwsg'], 'clf_type': 1})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'predict'})
        response = view(request, pk=1)
        response.render()
        self.assertEqual(response.status_code, 405)

    def test_predict_returns_bad_request_if_data_invalid(self):
        factory = APIRequestFactory()
        request = factory.post('/datasets/1/prepare_dataset', {'train_size': 60, 'test_size': 20})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'prepare_dataset'})
        response = view(request, pk=1)

        request = factory.post('/datasets/1/predict', {'vp_video_ids': ['KDxJlW6cxRk', 'NYhxaZXXwsg'], 'clf': 1})
        force_authenticate(request, user=user)
        view = DatasetViewSet.as_view({'post': 'predict'})
        response = view(request, pk=1)
        response.render()
        self.assertEqual(response.status_code, 400)


    '''VideoViewSet'''
    def test_get_videos_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/videos/')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'get': 'list'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '[{"id":102,"vp_video_id":"vpVideoId","title":"videoTitle",' +
                                           '"description":"description","comments":"comments","label":"label",' +
                                           '"dataset":3,"subset_type":""}]')

    def test_get_video_returns_success(self):
        factory = APIRequestFactory()
        request = factory.get('/videos/102')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=102)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":102,"vp_video_id":"vpVideoId","title":"videoTitle",' +
                                           '"description":"description","comments":"comments","label":"label",' +
                                           '"dataset":3,"subset_type":""}')

    def test_get_video_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.get('/videos/102')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=102)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_create_video_returns_success(self):
        factory = APIRequestFactory()
        request = factory.post('/videos/', {'vp_video_id': "vpVideoId", 'title': "videoTitle",
                                            'description': "description", 'comments': "comments",
                                            'label': "label", 'dataset': 1, 'subset_type': ""})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'post': 'create'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.content, '{"id":103,"vp_video_id":"vpVideoId","title":"videoTitle",' +
                                            '"description":"description","comments":"comments","label":"label",' +
                                            '"dataset":1,"subset_type":""}')

    def test_create_videos_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.post('/videos/', [{'vp_video_id': "vpVideoId", 'title': "videoTitle",
                                            'description': "description", 'comments': "comments",
                                            'label': "label", 'dataset': 1, 'subset_type': ""},
                                            {'vp_video_id': "vpVideoId", 'title': "videoTitle",
                                            'description': "description", 'comments': "comments",
                                            'label': "label", 'dataset': 1, 'subset_type': ""}], format='json')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'post': 'create'})
        response = view(request)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_update_video_returns_success(self):
        factory = APIRequestFactory()
        request = factory.put('/videos/101', {'vp_video_id': "updatedVpVideoId", 'title': "updatedVideoTitle",
                                             'description': "description", 'comments': "comments",
                                             'label': "label", 'dataset': 2, 'subset_type': ""})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'put': 'update'})
        response = view(request, pk=101)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":101,"vp_video_id":"updatedVpVideoId","title":"updatedVideoTitle",' +
                                            '"description":"description","comments":"comments","label":"label",' +
                                            '"dataset":2,"subset_type":""}')

        request = factory.put('/videos/101', {'vp_video_id': "updatedVpVideoId", 'title': "updatedVideoTitle",
                                             'description': "description", 'comments': "comments",
                                             'label': "label", 'dataset': 1, 'subset_type': ""})
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'put': 'update'})
        response = view(request, pk=101)
        response.render()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, '{"id":101,"vp_video_id":"updatedVpVideoId","title":"updatedVideoTitle",' +
                                            '"description":"description","comments":"comments","label":"label",' +
                                            '"dataset":1,"subset_type":""}')

    def test_update_video_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.put('/videos/101', {'vp_video_id': "updatedVpVideoId", 'title': "updatedVideoTitle",
                                             'description': "description", 'comments': "comments",
                                             'label': "label", 'dataset': 3, 'subset_type': ""})
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'put': 'update'})
        response = view(request, pk=101)
        response.render()
        self.assertEqual(response.status_code, 403)

        request = factory.put('/videos/100', {'vp_video_id': "updatedVpVideoId", 'title': "updatedVideoTitle",
                                             'description': "description", 'comments': "comments",
                                             'label': "label", 'dataset': 1, 'subset_type': ""})
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'put': 'update'})
        response = view(request, pk=100)
        response.render()
        self.assertEqual(response.status_code, 403)

    def test_delete_video_returns_success(self):
        factory = APIRequestFactory()
        request = factory.delete('/videos/101')
        user = User.objects.get(pk=1)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=101)
        self.assertEqual(response.status_code, 204)

        request = factory.delete('/videos/100')
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=100)
        self.assertEqual(response.status_code, 204)

    def test_delete_video_returns_forbidden(self):
        factory = APIRequestFactory()
        request = factory.delete('/videos/101')
        user = User.objects.get(pk=2)
        force_authenticate(request, user=user)
        view = VideoViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=101)
        self.assertEqual(response.status_code, 403)
