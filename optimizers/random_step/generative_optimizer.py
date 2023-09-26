import collections
import numpy as np
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings("ignore")


class GenerativeOptimizer:
    """
        At first, we generate neighborhood data by randomly perturbing features
        from the sample. We then find probabilities for each of them and choose the best one.
        Next, we iterate this algorithm for a new sample.
        All in all, we gain the best-valued sample with the best probability,
        but still close enough to the starting sample.
    """

    def __init__(self, model, training_data, mutable_cols, margin=0, weights=[]):
        """
        mutable_cols: list with columns' names.
        margin: the percent from max value in column which is added to min / max if needed.
                Must be given in decimal format, default = 0."""

        self.training_data = training_data
        self.iter_samples_ = []
        self.show_samples_ = []
        self.model = model
        self.mutable_cols = mutable_cols
        self.margin = margin
        self.weights = weights

    def __optimization(self, sample, mode, neighborhood, num_samples, desired_class, desired_proba):
        """
            Generates a new row sample geometrically close to a given one.

            Args:
                sample: 1d numpy array.
                neighborhood: number of the closest samples.
                              In fact, this value represents the proximity of random points.
                              If we want to find the best probability and we want our point to remain
                              as close as possible, then we can experiment with this parameter.
                              If it isn't our goal, then we can just set this to 1, default=1.
                num_samples: size of the neighborhood.

            Returns:
                A tuple with probability and 1d numpy array with values of a new sample.
        """

        # creating an array of needed shape
        data = np.array([sample] * num_samples)

        margins = np.zeros(len(self.mutable_cols)) if self.margin == 0 else self.margin
        weights_arr = np.ones(len(sample))

        mut_col_indexes = np.array([list(self.training_data.columns).index(self.mutable_cols[i])
                                    for i in range(len(self.mutable_cols))])

        # checking mutable columns for a type of values in it (discrete or continuous)
        for i in mut_col_indexes:
            is_discrete = (1. * self.training_data.iloc[:, i].nunique()
                           / self.training_data.iloc[:, i].count()) < 0.05
            if is_discrete:  # generating a distribution from probabilities of each unique value
                feature_count = collections.Counter(self.training_data.iloc[:, i])
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
                freqs = (np.array(frequencies) / float(sum(frequencies)))

                inverse_column = np.random.choice(values, size=num_samples, replace=True, p=freqs)

                data.T[i] = inverse_column
            else:  # generating values from uniform distribution
                col_min = self.training_data.iloc[:, i].min()
                col_max = self.training_data.iloc[:, i].max()
                mrgn = (col_max - col_min) * margins[i]
                inverse_column = np.random.uniform(col_min - mrgn,
                                                   col_max + mrgn,
                                                   num_samples)
                data.T[i] = inverse_column
                if len(self.weights) > 0:
                    weights_arr[i] *= self.weights[i]

        # adding our original sample to the first place,
        # because we don't want to choose a probability worth than initial
        data[0] = sample

        if mode == 'step':
            # finding distances from original to each generated
            distance_metric = 'euclidean'
            distances = pairwise_distances(
                data,
                np.array(sample).reshape(1, -1),
                metric=distance_metric
            ).ravel()

            # choosing the closest
            nearest_indexes = np.argsort(distances)[:neighborhood]
            probas = prob(self.model, data[nearest_indexes]).T[desired_class]
            closest_prob_index = list(abs(probas - desired_proba)).index(np.min(abs(probas - desired_proba)))
            best_sample = data[nearest_indexes][closest_prob_index]

            return probas[closest_prob_index], best_sample

        elif mode == 'formula':
            distances = np.sum(weights_arr * abs(data - np.array([self.initial_sample] * num_samples)), axis=1)
            # choosing the closest
            nearest_indexes = np.argsort(distances)[:neighborhood]
            probas = prob(self.model, data[nearest_indexes]).T[desired_class]
            formula = abs(probas - desired_proba) + np.sort(distances[:neighborhood])

            closest_prob_index = list(formula).index(np.min(formula))
            best_sample = data[nearest_indexes][closest_prob_index]

            return probas[closest_prob_index], best_sample
        else:
            raise ValueError('Parameter "mode" must be either "step" or "formula", not "{mode}".'.format(mode=mode))

    def optimize_sample(self,
                        sample,
                        mode,
                        neighborhood=1,
                        num_samples=5000,
                        desired_class=0,
                        desired_proba=1):
        """
            Iterates the generative algorithm.

            Args:
                sample: 1d numpy array.
                mode: a string specifying the method used in algorithm. Should be either 'step' or 'formula'.
                      Otherwise, WrongModeError is raised.
                neighborhood: number of the closest samples.
                              In fact, this value represents the proximity of random points,
                              default=1. If we want to approximate new points to initial one,
                              then we simply give to the neighborhood value of less then 1.
                              The closer this value to 0, the closer is neighborhood and the bigger
                              number of samples you receive unless you get the best one.
                num_samples: size of the neighborhood, default=5000.
                desired_class: class to optimize, represented by number. (model.classes_)
                desired_proba: value of maximum probability, if don't need more, default=1.

            Returns:
                A tuple with probability and 1d numpy array with values of a new sample.
        """

        self.initial_sample = sample
        neighborhood = int(neighborhood * num_samples)
        proba = prob(self.model, [self.initial_sample]).ravel()[desired_class]
        self.iter_samples_.append((proba, np.array(self.initial_sample)))
        _, best_sample = self.__optimization(sample,
                                             mode,
                                             num_samples,
                                             num_samples,
                                             desired_class,
                                             desired_proba)

        least_possible_diff = abs(prob(self.model, [best_sample]).ravel()[desired_class] - desired_proba)
        c = 0
        while abs(proba - desired_proba) > least_possible_diff:
            _, sample = self.__optimization(sample,
                                            mode,
                                            neighborhood,
                                            num_samples,
                                            desired_class,
                                            desired_proba)

            proba = prob(self.model, [sample]).ravel()[desired_class]
            if round(abs(proba - desired_proba), 2) < round(abs(self.iter_samples_[-1][0] - desired_proba), 2):
                self.iter_samples_.append((proba, sample))
            if c > 1000:
                break
            c += 1
        for _ in range(50):
            _, sample = self.__optimization(sample,
                                            mode,
                                            neighborhood,
                                            num_samples,
                                            desired_class,
                                            desired_proba)

            proba = prob(self.model, [sample]).ravel()[desired_class]
            if round(abs(proba - desired_proba), 2) < round(abs(self.iter_samples_[-1][0] - desired_proba), 2):
                self.iter_samples_.append((proba, sample))

        return proba, sample

    def show_samples(self):
        """
            The function shows suggested results of optimization based on self.iter_samples_.
        """

        length = len(self.iter_samples_)
        descriptive_dict = {'25%': self.iter_samples_[int(length * 0.25)],
                            'Mid-point': self.iter_samples_[int(length // 2)],
                            '75%': self.iter_samples_[int(length * 0.75)],
                            'Best point': self.iter_samples_[-1]}
        return descriptive_dict


def prob(model, data):
    return np.array(list(model.predict_proba(data)))
