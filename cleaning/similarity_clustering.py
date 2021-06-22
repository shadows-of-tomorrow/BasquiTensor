import os
import numpy as np
import cv2.cv2 as cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SimilarityClusterer:
    """
    Clusters images in a folder based on visual similarity.
    Visual similarity is defined here as activations of the VGG16 model.
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.n_clusters = 10
        self.similarity_model = self._construct_similarity_model()
        self.input_shape = self.similarity_model.input_shape[1:-1]

    def run(self):
        self._create_cluster_folders()
        image_names = os.listdir(self.input_path)
        features = self._extract_features(image_names)
        features = self._reduce_feature_dimensionality(features)
        image_clusters = self._cluster_feature_vectors(features)
        self._store_image_clusters(image_clusters, image_names)

    def _store_image_clusters(self, image_clusters, image_names):
        assert len(image_clusters) == len(image_names)
        for k in range(len(image_names)):
            try:
                image = self._read_image(image_names[k])
                self._write_image(image, image_clusters[k], image_names[k])
            except Exception as e:
                print(f"{e}")

    def _create_cluster_folders(self):
        for k in range(self.n_clusters):
            os.makedirs(f"{self.output_path}{k}")

    def _cluster_feature_vectors(self, features):
        k_means = KMeans(n_clusters=self.n_clusters, n_jobs=-1)
        k_means.fit(features)
        return list(k_means.labels_)

    def _reduce_feature_dimensionality(self, features, n_components=100):
        pca = PCA(n_components=n_components)
        pca.fit(features)
        return pca.transform(features)

    def _extract_features(self, image_names):
        features = []
        for image_name in image_names:
            try:
                image = self._read_image(image_name)
                image_features = self._extract_features_from_image(image)
                features += [image_features]
            except Exception as e:
                print(f"{e}")
        return np.stack(features, axis=0).reshape(-1, 4096)

    def _extract_features_from_image(self, image):
        image = self._reshape_image(image)
        image = preprocess_input(image)
        features = self.similarity_model.predict(image, use_multiprocessing=True)
        return features

    def _reshape_image(self, image):
        image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR)
        image = image.reshape(1, self.input_shape[0], self.input_shape[1], 3)
        return image

    def _write_image(self, image, image_cluster, image_name):
        cv2.imwrite(f"{self.output_path}{image_cluster}/{image_name}", image)

    def _read_image(self, image_name):
        image = cv2.imread(f"{self.input_path}{image_name}")
        return image

    @staticmethod
    def _construct_similarity_model():
        vgg16 = VGG16()
        model = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)
        return model
