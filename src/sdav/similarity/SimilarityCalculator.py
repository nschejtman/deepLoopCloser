import numpy as np


class SimilarityCalculator:
    def __init__(self, dataset: np.ndarray, mu=0.5, sigma=0.2, a=10, b=-10):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.dataset = dataset

    def similarity_score(self, h1, h2):
        average_response = self._average_response(self.dataset)
        distinctive_score = self._distinctive_score(average_response)
        matched_features = SimilarityCalculator._match_features(h1, h2)
        weighted_distances = SimilarityCalculator._compute_weighted_distances(matched_features, distinctive_score)
        return self._calculate_similarity_from_distances(weighted_distances)

    @staticmethod
    def _average_response(dataset):
        x0 = dataset.shape[0] * dataset.shape[1]
        x1 = dataset.shape[2]
        return np.average(dataset.reshape(x0, x1), axis=0)

    def _distinctive_score(self, h):
        a = - ((h - self.mu) ** 2) / (2 * self.sigma ** 2)
        return np.exp(a)

    @staticmethod
    def _match_features(m1, m2):
        matched_features = []
        for i, mi in enumerate(m1):
            absolute_distance = m2 - mi
            norms = np.linalg.norm(absolute_distance, axis=1)
            min_idx = np.argmin(norms)
            matched_features.append((mi, m2[min_idx]))
        return matched_features

    @staticmethod
    def _compute_weighted_distances(feature_matches, distinctive_score):
        def feature_match_weighted_distance(match):
            wd = np.matmul(distinctive_score, (match[0] - match[1]))
            return np.linalg.norm(wd)

        return list(map(feature_match_weighted_distance, feature_matches))

    def _calculate_similarity_from_distances(self, distances):
        similarities = list(map(lambda sk: self.a + self.b * np.log(sk), distances))
        return np.sum(similarities)
