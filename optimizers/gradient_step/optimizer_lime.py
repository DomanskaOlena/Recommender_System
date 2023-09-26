import numpy as np
import scipy as sp
# import pandas as pd
import collections
from copy import deepcopy
from optimizers.gradient_step.explainer import LimeTabularExplainer


def prob(model, data):
    return np.array(list(model.predict_proba(data)))


def random_sample_generator(data, shape):
    random_sample = np.zeros((1, shape))
    while prob(random_sample)[0][0] > prob(random_sample)[0][1]:
        for i in range(shape):
            inverse_column = np.random.uniform(data.iloc[:, i].min(),
                                               data.iloc[:, i].max(),
                                               1)
            random_sample.T[i] = inverse_column
    return random_sample


def prepare_data_prob(model, data, not_opt_data):
    not_opt_data_all_rows = np.tile(not_opt_data, (data.shape[0], 1))
    construct_data = np.concatenate((data, not_opt_data_all_rows), axis=1)
    # np.array(list(model.predict_proba(construct_data)))
    return prob(model, construct_data)


class OptimizerLime:
    """
        We simplify the input model, construct loss function using desired porbability.
        Then we use the gradient step to go in the right direction and get an optimized value.
    """
    def __init__(self, model, training_data, mutable_cols, margin=0, num_iterations=200, weights=[],
                 mode='simple'):

        self.training_data = training_data
        self.iter_samples_ = []
        self.model = model
        self.mutable_cols = mutable_cols
        self.margin = margin
        self.num_iterations = num_iterations
        self.weights = weights
        self.mode = mode

    def __optimization(self, sample, wanted_proba, wanted_class=0, num_samples=5000, detailed_mode=False):
        """
            Finding an optimized variables.

            Returns:
                A tuple of summaries and optimized values
        """
        # data preparation
        # creating an array of needed shape
        data = np.array([sample] * num_samples)

        margins = np.zeros((len(self.mutable_cols))
                           ) if self.margin == 0 else self.margin

        if len(self.weights) < 1:
            self.weights = np.ones(len(self.mutable_cols))

        # checking mutable columns for a type of values in it (discrete or continuous)
        for i in range(len(self.mutable_cols)):
            is_discrete = (1. * self.training_data[self.mutable_cols].iloc[:, i].nunique()
                           / self.training_data[self.mutable_cols].iloc[:, i].count()) < 0.05
            if is_discrete:  # generating a distribution from probabilities of each unique value
                feature_count = collections.Counter(
                    self.training_data[self.mutable_cols].iloc[:, i])
                values, frequencies = map(
                    list, zip(*(sorted(feature_count.items()))))
                freqs = (np.array(frequencies) / float(sum(frequencies)))

                inverse_column = np.random.choice(
                    values, size=num_samples, replace=True, p=freqs)

                data.T[list(self.training_data.columns).index(self.mutable_cols[i])] \
                    = inverse_column
            else:  # generating values from uniform distribution
                col_min = self.training_data[self.mutable_cols].iloc[:, i].min(
                )
                col_max = self.training_data[self.mutable_cols].iloc[:, i].max(
                )
                mrgn = (col_max - col_min) * margins[i]
                inverse_column = np.random.uniform(col_min - mrgn,
                                                   col_max + mrgn,
                                                   num_samples)
                data.T[list(self.training_data.columns).index(self.mutable_cols[i])] \
                    = inverse_column

        # adding our original sample to the first place,
        # because we don't want to choose a probability worse than initial
        data[0] = sample

        probas = prob(self.model, data)
        if self.mode == 'simple':

            min_val_ind = np.argmin(np.array([abs(probas_one[wanted_class] - wanted_proba)
                                              for probas_one in probas]))
            # list( probas[[abs(probas_one[wanted_class] - wanted_proba) == min_val for probas_one in probas]])
            nearest_prob_val = probas[min_val_ind]
            self.nearest_prob_val = nearest_prob_val

            summary, best_optimized_val = self.optimize_sample_gradient(sample[len(self.mutable_cols):].values,
                                                                        sample[:len(self.mutable_cols)], self.simple_abs, 
                                                                        goal_certainty=wanted_proba, detailed=detailed_mode)
        elif self.mode == 'formula':

            summary, best_optimized_val = self.optimize_sample_gradient(sample[len(self.mutable_cols):].values,
                                                                        sample[:len(self.mutable_cols)], self.formula_using_distance, 
                                                                        goal_certainty=wanted_proba, detailed=detailed_mode)
        return summary, best_optimized_val

    @staticmethod
    def gradient_step(start_val, weights, cur_prob, desired_prob, step=0.1):
        ret_val = start_val
        for i in range(len(weights)):
            ind, weigh_val = weights[i]
            ret_val[ind] -= step * 2 * (cur_prob - desired_prob) * weigh_val
        return ret_val

    def optimize_sample(self, sample, wanted_class, desired_proba, detailed_mode=False, num_samples=5000):

        return self.__optimization(sample, desired_proba, wanted_class, num_samples, detailed_mode=detailed_mode)

    def simple_abs(self, desired_certainty, cur_certainty, inp_val=[], cur_val=[]):
        return abs(desired_certainty - cur_certainty)

    def formula_using_distance(self, desired_certainty, cur_certainty, inp_val, cur_val):
        prob_diff = self.simple_abs(desired_certainty, cur_certainty)
        distance__ = np.sqrt(
            np.sum(self.weights * abs(inp_val - np.array(cur_val)), )) / 10
        # / abs(inp_val))
        distance = np.sum(self.weights * abs(inp_val - np.array(cur_val)))
        return prob_diff + distance / 10

    def formula_for_explainer(self, model, data, not_opt_data, desired_certainty, inp_val):
        not_opt_data_all_rows = np.tile(not_opt_data, (data.shape[0], 1))
        construct_data = np.concatenate((data, not_opt_data_all_rows), axis=1)
        probability = prob(model, construct_data)
        res = [[self.formula_using_distance(
            desired_certainty, probability[i, :][0], inp_val, data[i]), self.formula_using_distance(
            1 - desired_certainty, probability[i, :][1], inp_val, data[i])] for i in range(data.shape[0])]

        res = np.array(res)
        res = res / res.sum(axis=1)[:, None]
        return res

    def optimize_sample_gradient(self, not_opt_val, inp_val, formula_func, goal_certainty=0.6, step=1,
                                 target_class=0, prob_func=prepare_data_prob, detailed=False):
        """
            Optimizing the variables using gradient step

        Args:
            not_opt_val (pd.DataFrame): values that won't be optimized
            inp_val (pd.DataFrame): values that can be optimized
            formula_func (function): formula for comparing the dots 
            goal_certainty (float, optional): the desired score of the class. Defaults to 0.6.
            step (int, optional): learning rate. Defaults to 1.
            target_class (int, optional): class we trying to optimize. Defaults to 0.
            prob_func (function, optional): function for calculating probability. Defaults to prepare_data_prob.
            detailed (bool, optional): if True gives more information regarding the steps. Defaults to False.

        Returns:
            list[list[], list[]], list[float, list[float]]: 
            summary - general steps the approach did
            improving_summary - improving steps the approach did
            most_close_val_feat - [probability of desired class, features]
        """
        prob_pred = prob_func(self.model, np.array([inp_val]), not_opt_val)
        summary = [[prob_pred, inp_val]]
        improving_summary = [[prob_pred[0][target_class], inp_val]]

        explainer = LimeTabularExplainer(np.array(self.training_data[self.mutable_cols].values),
                                         self.model, mode='classification', feature_names=self.mutable_cols)
        start_inp_val = deepcopy(inp_val)

        exp = explainer.explain_instance_find_optimal(inp_val, prob_func, num_features=2,
                                                      not_optimize_data=not_opt_val)
        if detailed:
            print(f'Starting probability {prob_pred[0]}')
        most_close_iter = 0
        # abs(goal_certainty - prob_pred[0][target_class])
        most_close_val = formula_func(
            goal_certainty, prob_pred[0][target_class], start_inp_val, start_inp_val)
        most_close_val_feat = [prob_pred[0][target_class], inp_val]
        changed_step = 0
        for iii in range(self.num_iterations):
            
            prob_prev = prob_func(self.model, np.array([inp_val]), not_opt_val)[0][target_class]
            opt_val = self.gradient_step(inp_val, exp.local_exp[1], prob_prev, goal_certainty, step)
            prob_pred = prob_func(self.model, np.array([opt_val]), not_opt_val)
            summary.append([deepcopy(prob_pred), deepcopy(opt_val)])

            cur_close_val = formula_func(
                goal_certainty, prob_pred[0][target_class], start_inp_val, opt_val[:start_inp_val.shape[0]])
            if most_close_val > cur_close_val:
                most_close_val = cur_close_val
                most_close_iter = 0
                most_close_val_feat = [deepcopy(prob_pred[0][target_class]), deepcopy(opt_val)]
                exp = explainer.explain_instance_find_optimal(inp_val, prob_func, num_features=2,
                                                          not_optimize_data=not_opt_val)
                improving_summary.append(deepcopy(most_close_val_feat))
            else:
                most_close_iter += 1

            if (changed_step > min(self.num_iterations / 100, 5)):
                break

            inp_val = deepcopy(opt_val)
            if (most_close_iter > min(self.num_iterations / 20, 10)):
                changed_step += 1
                step /= 10
                most_close_iter = 0
                if detailed:
                    print(f'Decreased step to {step}.')
                inp_val = most_close_val_feat[1]
                exp = explainer.explain_instance_find_optimal(inp_val, prob_func, num_features=2,
                                                          not_optimize_data=not_opt_val)

            if (iii % 25 == 0) and detailed:
                print(
                    f'iteration {iii}: the closest probability {most_close_val_feat[0]}')
            if iii % 3 == 0: # to decrease the time, we rerun the explainer every 3rd time
                exp = explainer.explain_instance_find_optimal(inp_val, prob_func, num_features=2,
                                                          not_optimize_data=not_opt_val)
        most_close_val_feat[1] = np.append(most_close_val_feat[1], not_opt_val)
        return [summary, improving_summary], most_close_val_feat
