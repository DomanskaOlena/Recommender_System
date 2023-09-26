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

    def __init__(self, model, training_data, mutable_cols, margin=0):
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

    def __optimization(self, sample, neighborhood, num_samples, desired_class, desired_proba):
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

        margins = np.zeros((1, len(self.mutable_cols))) if self.margin == 0 else self.margin

        # checking mutable columns for a type of values in it (discrete or continuous)
        for i in range(len(self.mutable_cols)):
            is_discrete = (1. * self.training_data[self.mutable_cols].iloc[:, i].nunique()
                           / self.training_data[self.mutable_cols].iloc[:, i].count()) < 0.05
            if is_discrete:  # generating a distribution from probabilities of each unique value
                feature_count = collections.Counter(self.training_data[self.mutable_cols].iloc[:, i])
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
                freqs = (np.array(frequencies) / float(sum(frequencies)))

                inverse_column = np.random.choice(values, size=num_samples, replace=True, p=freqs)

                data.T[list(self.training_data.columns).index(self.mutable_cols[i])] \
                    = inverse_column
            else:  # generating values from uniform distribution
                col_min = self.training_data[self.mutable_cols].iloc[:, i].min()
                col_max = self.training_data[self.mutable_cols].iloc[:, i].max()
                mrgn = (col_max - col_min) * margins[i]
                inverse_column = np.random.uniform(col_min - mrgn,
                                                   col_max + mrgn,
                                                   num_samples)
                data.T[list(self.training_data.columns).index(self.mutable_cols[i])] \
                    = inverse_column

        # adding our original sample to the first place,
        # because we don't want to choose a probability worth than initial
        data[0] = sample

        # finding distances from original to each generated
        distance_metric = 'euclidean'
        distances = pairwise_distances(
            data,
            np.array(sample).reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # choosing the closest
        nearest_index = np.argsort(distances)[:neighborhood]

        probas = prob(self.model, data[nearest_index]).T[desired_class]

        if desired_proba == 1:
            max_prob_indx = np.argsort(probas)[-1]
            best_sample = data[nearest_index][max_prob_indx]

            return probas[max_prob_indx], best_sample
        else:
            max_prob_indx = list(probas).index(np.max(probas[probas <= desired_proba]))
            best_sample = data[nearest_index][max_prob_indx]

            return probas[max_prob_indx], best_sample

    def optimize_sample(self,
                        sample,
                        neighborhood=1,
                        num_samples=30000,
                        desired_class=0,
                        desired_proba=1):
        """
            Iterates the generative algorithm.

            Args:
                sample: 1d numpy array.
                neighborhood: number of the closest samples.
                              In fact, this value represents the proximity of random points,
                              default=1. If we want to approximate new points to initial one,
                              then we simply give to the neighborhood value of less then 1.
                              The closer this value to 0, the closer is neighborhood and the bigger
                              number of samples you receive unless you get the best one.
                num_samples: size of the neighborhood, default=30000.
                desired_class: class to optimize, represented by number. (model.classes_)
                desired_proba: value of maximum probability, if don't need more, default=1.

            Returns:
                A tuple with probability and 1d numpy array with values of a new sample.
        """

        neighborhood = int(neighborhood * num_samples)
        proba = prob(self.model, [sample])[0][desired_class]
        self.iter_samples_.append((proba, np.array(sample)))

        best_possible_proba, _ = self.__optimization(sample,
                                                     num_samples,
                                                     num_samples,
                                                     desired_class,
                                                     desired_proba)

        c = 0
        while proba < best_possible_proba:
            proba, sample = self.__optimization(sample,
                                                neighborhood,
                                                num_samples,
                                                desired_class,
                                                desired_proba)

            if round(proba, 2) > round(self.iter_samples_[-1][0], 2):
                self.iter_samples_.append((proba, sample))

            if c == 100 or c == 250 or c == 400:
                neighborhood = neighborhood + int(num_samples * 0.1)
                print('0.1 added to neighborhood')
            elif c > 500:
                break

            c += 1

        return proba, sample

    def show_samples(self):
        """
            The function shows suggested results of optimization based on self.iter_samples_.
        """

        length = len(self.iter_samples_)
        descriptive_dict = {'25%': self.iter_samples_[int(length*0.25)],
                            'Mid-point': self.iter_samples_[int(length//2)],
                            '75%': self.iter_samples_[int(length*0.75)],
                            'Best point': self.iter_samples_[-1]}
        return descriptive_dict


def prob(model, data):
    return np.array(list(model.predict_proba(data)))
