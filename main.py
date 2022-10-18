# -*- coding: utf-8 -*-
r"""Module for feature optimizer"""
import copy
import logging
import random
import shelve
from multiprocessing import Manager, Process
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from commons.estimators import EstimatorType

LOGGER = logging.getLogger(__name__)

SyncDictType = Dict[str, Dict[str, Union[None, bool, float, str, pd.Series]]]


class BestFeatureGrid:
    r"""
    Class which can select the best set of features for existing model
    through score maximization
    """

    def __init__(self, estimator: EstimatorType, features: pd.DataFrame, labels: pd.Series, groups: pd.Series,
                 scorer: _BaseScorer, result_folder: Union[str, Path], n_jobs: int = 1,
                 cv: Union[int, BaseCrossValidator, None] = None, init_prob: Optional[pd.Series] = None,
                 start_features: Optional[pd.Series] = None, decision_threshold: float = 0.85,
                 poisson_range_true: float = 3, poisson_range_false: float = 2, learning_rate: float = 0.01,
                 limit_iterations: int = 9999, greater_score_is_better: bool = True):
        """
        Args:
            estimator: An estimator object implementing ‘fit’ which will provide score for comparison different
                subsets of features.
            features: DataFrame with features.
            labels: Series with labels.
            groups: Series used for grouping features in cross validation.
            scorer: A scorer, based on _BaseScorer from Sklearn.
            result_folder: Path to folder where results will be store.
            n_jobs: Amount of independent explorers of feature space at once, each uses his own cpu core.
            cv: A number of splits in cross validation or a custom cross validation with specific rules.
            start_features: A series with start subset of features, it's empty by default and filling by optimizer with
                useful features during the work
            init_prob: A series, equals to features by its width, but with feature names in indexes
                and their probabilities whose sum is 1 in values.
            decision_threshold: A threshold of making decision for each explorer. He will decide to explore a new part
                of feature space or recheck known one.
            poisson_range_true: It affects on amount of features, that we will pick for another experiment.
                It reflects the meaning of the word `a little`.
            poisson_range_false: It affects on amount of features, that we will try to remove from another experiment.
                It reflects the meaning of the word `a bit`.
            learning_rate: Multiplier of updating probabilities speed.
            limit_iterations: Amount of iterations that the optimizer will live.
            greater_score_is_better: If True set maximize score in optimizer
        """
        self.estimator = estimator
        self.features_df = features.loc[:, features.columns.sort_values()]
        self.labels = labels
        self.groups = groups
        self.cv = cv
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.result_folder = Path(result_folder)
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.limit_iterations = limit_iterations
        self.greater_score_is_better = greater_score_is_better

        # optimizer state
        self.iteration: int = 0
        self.total_steps: int = 0
        self.last_step: int = 0
        self.best_score: float = np.NINF
        self.best_feature_importances = start_features if start_features is not None else pd.Series()

        prob_values: Union[np.ndarray, List[float]]
        if init_prob is not None:
            prob_values = [init_prob[col] if col in init_prob else init_prob.min() for col in self.features_df.columns]
        else:
            prob_values = np.ones(len(self.features_df.columns))
        self.probabilities = self.normalize_probabilities(pd.Series(prob_values, index=self.features_df.columns))
        self.sync_dict: SyncDictType = Manager().dict()

        # params
        self.decision_threshold = decision_threshold
        self.poisson_range_true = poisson_range_true
        self.poisson_range_false = poisson_range_false
        self.learning_rate = learning_rate

        # TODO: check for (self.features_df, self.labels, self.groups) have same lengths

    @property
    def best_features(self) -> Set[str]:
        r"""Property for set of best features"""
        return set(self.best_feature_importances.index)

    @staticmethod
    def normalize_probabilities(probabilities: pd.Series) -> pd.Series:
        r"""Method which keeps sum of all probabilities equals to 1"""
        probabilities = probabilities.clip(np.finfo(float).tiny, 1)
        probabilities = probabilities / probabilities.sum()
        return probabilities

    def calc_prob_delta(self, feature_importances: pd.Series, score_delta: float) -> pd.Series:
        r"""Calculate the reward or the punishment for selected features"""
        gradient: pd.Series = feature_importances.fillna(feature_importances.min() / 100.0) * score_delta
        gradient = gradient.apply(lambda f: np.abs(np.tanh(f * 0.1))) * self.learning_rate
        gradient = gradient * np.sign(score_delta)
        return gradient

    def validate_model(self, features: List[str]) -> float:
        r"""Returns the score of selected features set"""
        model = copy.deepcopy(self.estimator)
        local_features_df = self.features_df.loc[:, features]
        scores = cross_val_score(estimator=model, X=local_features_df.values, y=self.labels.values, cv=self.cv,
                                 scoring=self.scorer, groups=self.groups)
        return np.array(scores).mean()

    def refit_model(self, features: List[str]) -> pd.Series:
        r"""Fit model on whole features to get it's importances"""
        model = copy.deepcopy(self.estimator)
        local_features_df = self.features_df[features]
        model.fit(X=local_features_df.values, y=self.labels.values)
        return pd.Series(model.feature_importances_, index=local_features_df.columns)

    def get_features_for_validation(self) -> Tuple[List[str], List[str]]:
        r"""Select random new sets for validation for each worker per iteration"""
        # decision about add(True) or remove(False) some features from best collection of features between iterations
        decision = bool(np.random.binomial(1, self.decision_threshold)) if len(self.best_features) > 3 else True
        # set decision as False(remove) if size of best collection equal to size of all features
        decision = decision if len(self.best_features) < self.features_df.shape[1] else False
        updated_features: Set[str]
        selected_features: Set[str]
        if decision:
            feature_amount = np.random.poisson(self.poisson_range_true) + 1
            local_probabilities = self.normalize_probabilities(self.probabilities.drop(self.best_features))
            selected_features = set(np.random.choice(local_probabilities.index.values,
                                                     size=min(feature_amount, len(local_probabilities)),
                                                     replace=False, p=local_probabilities.values))
            updated_features = self.best_features.union(selected_features)
        else:
            feature_amount = np.random.poisson(self.poisson_range_false) + 1
            local_probabilities = self.normalize_probabilities(self.probabilities[self.best_features])
            updated_features = set(np.random.choice(local_probabilities.index.values,
                                                    size=max(len(self.best_features) - feature_amount, 3),
                                                    replace=False, p=local_probabilities.values))
            selected_features = self.best_features.difference(updated_features)

        return list(sorted(updated_features)), list(sorted(selected_features))

    def explore_vicinity(self, pid: int, sync_dict: SyncDictType) -> None:
        r"""Explore the vicinity of the current state of optimizer"""
        try:
            random_state = int(f"{self.iteration}{pid}")
            np.random.seed(random_state)
            random.seed(random_state)
            updated_features, selected_features = self.get_features_for_validation()
            random.shuffle(updated_features)
            random.shuffle(selected_features)
            score = self.validate_model(features=updated_features)
            feature_importances = self.refit_model(features=updated_features)
            score_delta = score - self.best_score

            if set(selected_features).issubset(self.best_features):
                # if decision was to remove some features
                selected_importances = self.best_feature_importances.loc[selected_features]
                gradient = self.calc_prob_delta(feature_importances=selected_importances, score_delta=score_delta) * -1
            else:
                # if decision was to add some features
                selected_importances = feature_importances.loc[selected_features]
                gradient = self.calc_prob_delta(feature_importances=selected_importances, score_delta=score_delta)
            sync_dict[str(pid)] = {'gradient': gradient, 'score': score, 'feature_importances': feature_importances,
                                   'random_state': random_state, 'status': True, 'error': None}
        except Exception as err:
            sync_dict[str(pid)] = {'status': True, 'error': str(err)}
            raise AttributeError(err)
        self.print_status()

    def sync_results(self, sync_dict: SyncDictType) -> None:
        r"""Synchronize the explorers and combine their results to choose the best one and update probabilities"""
        self.last_step += 1
        gradients = []
        for pid, result in sync_dict.items():
            if result['error'] is not None or not isinstance(result['score'], float):
                print(f'\n\n\tIn process {pid} an error happened:\n\t\t{result["error"]}')
                continue
            gradients.append(result['gradient'])
            process_score: float = result['score']
            if self.greater_score_is_better:
                is_score_better = process_score > self.best_score
            else:
                is_score_better = process_score < self.best_score
            if is_score_better and isinstance(result['feature_importances'], pd.Series):
                self.best_score = process_score
                self.best_feature_importances = result['feature_importances']
                self.total_steps += 1
                self.last_step = 0
                with shelve.open(str(self.result_folder / 'features_history.shl')) as shl:
                    shl[f"rs_{result['random_state']}"] = {'iteration': self.iteration,
                                                           'pid': pid,
                                                           'score': process_score,
                                                           'feature_importances': result['feature_importances']}

        union_gradient = pd.concat(gradients, axis='columns').mean(axis='columns')
        self.probabilities.loc[union_gradient.index] += int(self.greater_score_is_better) * union_gradient
        self.probabilities = self.normalize_probabilities(probabilities=self.probabilities)
        with shelve.open(str(self.result_folder / 'prob_history.shl')) as shl:
            shl[f"i_{self.iteration}"] = self.probabilities
        log_message = f"\tscore: {self.best_score:.5f}, total_steps: {self.total_steps}, last_step: {self.last_step}"
        print(log_message, end='\n')
        LOGGER.info(f"Iteration: {self.iteration}" + log_message)

    def print_status(self) -> None:
        r"""Print the progress of optimization"""
        bar = ''.join(['▓' if pr['status'] else '░' for pr in self.sync_dict.values()])
        percent = np.floor((len([p for p in self.sync_dict.values() if p['status']]) / len(self.sync_dict)) * 100)
        print(f"\rIteration {self.iteration}: {bar} {int(percent)}%", end='', flush=True)

    def optimize(self) -> None:
        r"""For huge amount of iterations spawn a bunch of explorers which validate
        random subsets of provided features based on it's probabilities"""
        # TODO: dont spawn subprocesses if n_jobs=1
        while self.iteration < self.limit_iterations:
            self.iteration += 1
            self.sync_dict = Manager().dict()
            processes = []
            for pid in range(self.n_jobs):
                self.sync_dict[str(pid)] = {'status': False, 'error': None}
                process = Process(target=self.explore_vicinity, kwargs={'pid': pid, 'sync_dict': self.sync_dict})
                processes.append(process)
                process.start()
            self.print_status()
            for pid, process in enumerate(processes):
                process.join()
            for process in processes:
                process.kill()

            self.sync_results(sync_dict=self.sync_dict)
            del self.sync_dict
