from sets import Set
import random
import logging
import sys
from apiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from videos.models import Video
from videos.models import ScikitDataset
from videos.models import Analysis
from videos.models import Analyses
from videos.serializers import VideoSerializer
from videos.serializers import DatasetSerializer
from videos import settings



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MachineLearningManager():

    def prepare_dataset(self, dataset, train_size, test_size):
        logger.info("Called MachineLearningManager prepare_dataset with" +
                    " train_size: " + str(train_size) +
                    " test_size: " + str(test_size))

        self.set_dataset_is_prepared(dataset, False)

        videos_queryset = dataset.videos.all()
        video_serializer = VideoSerializer(videos_queryset, many=True)
        videos = video_serializer.data

        if self.__are_sizes_valid(videos, train_size, test_size):
            free_indexes = range(0, len(videos))
            rand_train_indexes = random.sample(free_indexes, train_size)
            self.__assign_subset_type_to_video(rand_train_indexes, 'train', videos, videos_queryset)

            free_indexes = list(set(free_indexes) - set(rand_train_indexes))
            rand_test_indexes = random.sample(free_indexes, test_size)
            self.__assign_subset_type_to_video(rand_test_indexes, 'test', videos, videos_queryset)

            free_indexes = list(set(free_indexes) - set(rand_test_indexes))
            validate_size = len(videos) - (train_size + test_size)
            rand_validate_indexes = random.sample(free_indexes, validate_size)
            self.__assign_subset_type_to_video(rand_validate_indexes, 'validate', videos, videos_queryset)

            self.set_dataset_is_prepared(dataset, True)
        else:
            logger.error('Subset sizes are not valid')
            raise ValueError('Subset sizes are not valid')

    def set_dataset_is_prepared(self, dataset, is_prepared):
        logger.info("Called MachineLearningManager set_dataset_is_prepared")
        if dataset.is_prepared is not is_prepared:
            data = {'is_prepared': is_prepared}
            serializer = DatasetSerializer(instance=dataset, data=data, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()

    def validate_analyse(self, dataset):
        logger.info("Called MachineLearningManager validate_analyse")
        if dataset.is_prepared:
            train_videos = self.__getVideosBySubsetType(dataset, 'train')
            validate_videos = self.__getVideosBySubsetType(dataset, 'validate')
            test_videos = self.__getVideosBySubsetType(dataset, 'test')

            clf_types = [0, 1];
            analyses = {}
            scikit_learn = ScikitLearn()
            scikit_dataset = ScikitDataset(train_videos, validate_videos, test_videos)
            for clf_type in clf_types:
                self.__train(clf_type, scikit_dataset, scikit_learn)
                validate_data = scikit_dataset.validate.get('data')
                predicted = scikit_learn.predict(validate_data)
                validate_target = scikit_dataset.validate.get('target')
                analysis = self.__analyse(predicted,
                                          validate_target,
                                          scikit_dataset.validate.get('target_names'),
                                          scikit_learn)
                analyses[clf_type] = analysis
        else:
            error_msg = "Dataset " + str(dataset.id) + " is not prepared"
            logger.error(error_msg)
            raise Exception(error_msg)

        return Analyses(analyses[0], analyses[1])


    def test_analyse(self, clf_type, dataset):
        logger.info("Called MachineLearningManager test_analyse")
        if dataset.is_prepared:
            train_videos = self.__getVideosBySubsetType(dataset, 'train')
            validate_videos = self.__getVideosBySubsetType(dataset, 'validate')
            test_videos = self.__getVideosBySubsetType(dataset, 'test')

            scikit_learn = ScikitLearn()
            scikit_dataset = ScikitDataset(train_videos, validate_videos, test_videos)
            self.__train(clf_type, scikit_dataset, scikit_learn)

            test_data = scikit_dataset.test.get('data')
            predicted = scikit_learn.predict(test_data)
            test_target = scikit_dataset.test.get('target')
            analysis = self.__analyse(predicted,
                                      test_target,
                                      scikit_dataset.test.get('target_names'),
                                      scikit_learn)
        else:
            error_msg = "Dataset " + str(dataset.id) + " is not prepared"
            logger.error(error_msg)
            raise Exception(error_msg)
        return analysis

    def predict(self, clf_type, dataset, vp_video_ids):
        logger.info("Called MachineLearningManager predict")
        if dataset.is_prepared:
            video_data_extractor = VideoDataExtractor(
                                       YoutubeVideoSearcher(
                                           build(settings.YOUTUBE_API_SERVICE_NAME,
                                               settings.YOUTUBE_API_VERSION,
                                               developerKey=settings.DEVELOPER_KEY)))
            extreacted_test_videos = video_data_extractor.extractByIds(vp_video_ids)
            serializer = VideoSerializer(extreacted_test_videos, many=True)

            train_videos = self.__getVideosBySubsetType(dataset, 'train')
            test_videos = serializer.data

            scikit_learn = ScikitLearn()
            scikit_dataset = ScikitDataset(train_videos, [], test_videos)
            self.__train(clf_type, scikit_dataset, scikit_learn)

            test_data = scikit_dataset.test.get('data')
            predicted_nums = scikit_learn.predict(test_data)

            predicted = []
            target_names = scikit_dataset.test.get('target_names')
            for predicted_num in predicted_nums:
                predicted.append(target_names[predicted_num])
        else:
            error_msg = "Dataset " + str(dataset.id) + " is not prepared"
            logger.error(error_msg)
            raise Exception(error_msg)

        return predicted

    def __analyse(self, predicted, target, target_names, scikit_learn):
        analysis = Analysis(scikit_learn.calculate_accuracy(predicted, target),
                            scikit_learn.prepare_classification_report(predicted, target, target_names),
                            scikit_learn.prepare_confusion_matrix(predicted, target, target_names))
        return analysis

    def __train(self, clf_type, scikit_dataset, scikit_learn):
        logger.info("Called MachineLearningManager train with" + " clf_type: " + str(clf_type))
        train_data = scikit_dataset.train.get('data')
        train_target = scikit_dataset.train.get('target')
        scikit_learn.train_classifier(clf_type, train_data, train_target)

    def __getVideosBySubsetType(self, dataset, subset_type):
        videos_queryset = dataset.videos.filter(subset_type=subset_type)
        serializer = VideoSerializer(videos_queryset, many=True)
        videos = serializer.data
        return videos

    def  __are_sizes_valid(self, videos, train_size, test_size):
        valid = False
        dataset_size = len(videos)
        if (train_size + test_size) <= dataset_size:
            valid = True
        return valid

    #TODO: very slow
    def __assign_subset_type_to_video(self, rand_indexes, subset_type, videos, videos_queryset):
        for rand_index in rand_indexes:
            video = videos[rand_index]
            videoObject = videos_queryset.get(pk=video['id'])
            serializer = VideoSerializer(instance=videoObject, data={'subset_type': subset_type}, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()


class ScikitLearn():

    def __init__(self):
        self.classifier = None

    def train_classifier(self, clf_type, data, target):
        if clf_type == 0:
            self.classifier = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), min_df=1)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB()),
            ])
        elif clf_type == 1:
            self.classifier = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), min_df=1)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=1e-3, n_iter=5, random_state=42)),
            ])

        self.classifier = self.classifier.fit(data, target)

    def predict(self, new_data):
        predicted = self.classifier.predict(new_data)
        return predicted

    def calculate_accuracy(self, predicted, target):
        accuracy = str(np.mean(predicted == target)) + "\n"
        return accuracy

    def prepare_classification_report(self, predicted, target, target_names):
        return metrics.classification_report(target, predicted, target_names=target_names)

    def prepare_confusion_matrix(self, predicted, target, target_names):
        predictedStr = []
        targetStr = []
        for predicted_single, target_single in zip(predicted, target):
            predictedStr.append(target_names[predicted_single])
            targetStr.append(target_names[target_single])

        return pd.crosstab(pd.Series(targetStr), pd.Series(predictedStr), rownames=['Actual'], colnames=['Predicted'], margins=True)


class VideoDataExtractor():

    def __init__(self, youtube_video_searcher):
        self.youtube_video_searcher = youtube_video_searcher

    def extract(self, query):
        logger.info('Called VideoDataExtractor extract with query: ' + str(query))
        return self.youtube_video_searcher.search(query)

    def extractByIds(self, vp_video_ids):
        logger.info('Called VideoDataExtractor extractByIds')
        return self.youtube_video_searcher.searchByIds(vp_video_ids)


class YoutubeVideoSearcher():

    def __init__(self, youtube):
        self.youtube = youtube

    def searchByIds(self, vp_video_ids):
        videos = []
        for vp_video_id in vp_video_ids:
            response = self.youtube.videos().list(
                part="id,snippet",
                id=vp_video_id
            ).execute()
            self.__retrieveMetaDataFromResponseSingle(response, videos)

        return videos

    def search(self, query):
        videos = []
        response = self.youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults="10"
        ).execute()

        self.__retrieveMetaDataFromResponse(response, videos)

        #TODO: Write paging

        while('nextPageToken' in response):
            next_page_token = response['nextPageToken']
            response = self.__getNextPage(next_page_token, query)
            self.__retrieveMetaDataFromResponse(response, videos)

        logger.info("YoutubeVideoSearcher number of videos found: " + str(len(videos)))
        return videos


    def __retrieveMetaDataFromResponse(self, response, videos):
        for item in response.get("items", []):
            video = Video(
                vp_video_id=item["id"]["videoId"],
                title=item["snippet"]["title"],
                description=item["snippet"]["description"],
                comments="",
                label="",
            )
            videos.append(video)

    def __retrieveMetaDataFromResponseSingle(self, response, videos):
        for item in response.get("items", []):
            video = Video(
                vp_video_id=item["id"],
                title=item["snippet"]["title"],
                description=item["snippet"]["description"],
                comments="",
                label="",
            )
            videos.append(video)

    def __getNextPage(self, next_page_token, query):
        response = self.youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults="10",
            pageToken=next_page_token
        ).execute()

        return response
