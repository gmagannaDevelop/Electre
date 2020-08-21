"""
    ELECTRE :  Multi-criteria decision making
"""
import numpy as np
import pandas as pd
import subprocess

from functools import reduce, partial
from toolz.curried import compose
from sklearn import preprocessing
from typing import Dict, List, NoReturn, Callable, Any, Tuple, Union, Optional

import matplotlib.pyplot as plt
import seaborn as sns

class Electre(object):
    """
    Params :
        table: pandas dataframe, containing 
                alternatives as rows
                criteria as columns

      weights: dict { "criterion": [0,1] } 
                    the sum over all criteria must be equal to one

     criteria: dictionary { "criterion": 0 or 1 }
                            0 := minimise
                            1 := maximise

    threshold: threshold, defaults to 1.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        weights: Dict[str,float],
        criteria: Dict[str,int],
        threshold: Optional[float] = None
    ):
        self.table = table
        self.weights = weights
        self.criteria = criteria
        self.threshold = threshold or 0.5
    ##

    def fit(self):
        self.n_table = Electre._normalize_matrix(self.table, self.criteria)
        # create the weighted normalised decision matrix.
        self.w_n_table = self.n_table.copy()
        for column in self.n_table.columns:
            self.w_n_table[column] = self.n_table[column].map(lambda x: x*self.weights[column])

        # Create concordance and discordance matrices :
        self.c_matrix = Electre._concordance_matrix(self.n_table, self.weights)
        self.d_matrix = Electre._discordance_matrix(self.w_n_table)

        # Create the concordant dominance matrix :
        cdm_filter = partial(lambda x, y: 1 if y > x else 0, self.threshold)
        self.c_d_matrix = self.c_matrix.applymap(cdm_filter)

        # Create the discordant dominance matrix :
        self.d_threshold = self.d_matrix.mean().mean()
        cdm_filter = partial(lambda x, y: 1 if y < x else 0, self.d_threshold)
        self.d_d_matrix = self.d_matrix.applymap(cdm_filter)

        # Agregated dominance matrix :
        self.adm = Electre._agregated_dominance_matrix(self.c_d_matrix, self.d_d_matrix)

        # Results as a count of dominance
        self.dominance_rank = self.adm.sum(axis=1)
        self.submission_rank = self.adm.sum(axis=0)


    def plot_dominance(self):
        """ """
        self.dominance_rank.plot()

    def plot_submission(self):
        """ """
        self.submission_rank.plot()

    @staticmethod
    def save_as_image(df: pd.core.frame.DataFrame, name: str):
        df.to_html(name)
        subprocess.call(f'wkhtmltoimage -f png --width 0 {name} {name}.png', shell=True)
    ##

    @staticmethod
    def _normalize_matrix(
        table: pd.core.frame.DataFrame,
        direction: Dict[str, int],
        normalisation_rule: int = 2
    ) -> NoReturn:
        """
            Read normalisation part of the notebook to understand the rules.
        """
        if normalisation_rule < 1 or normalisation_rule > 4:
            raise Exception('Please specify a normalisation rule i from the interval [1,4]')

        n_table = table.copy()

        if normalisation_rule == 1:
            sq_sum_squares = table.apply(lambda y: np.sqrt(sum(x**2 for x in y)))
            sq_sum_squares = dict(sq_sum_squares)

            for column in table.columns:
                f = (lambda y: lambda x: x/sq_sum_squares[y])(column)
                n_table[column] = table[column].map(f)
            print(f'WARNING: normalisation rule {normalisation_rule} does not take into consideration MIN/MAX criteria')

        elif normalisation_rule == 2:
            denom = dict(table.copy().apply(lambda x: x.max() - x.min()))
            _min  = dict(table.copy().apply(lambda x: x.min()))
            _max  = dict(table.copy().apply(lambda x: x.max()))

            norm_min = lambda _key, element: (_max[_key] - element) / denom[_key]
            norm_max = lambda _key, element: (element - _min[_key]) / denom[_key]

            for column in table.columns:
                if direction[column]: # if True (==1), use the maximisation rule.
                    n_table[column] = table[column].map(partial(norm_max, column))

                else: # if False (==0), use the minimisation rule.
                    n_table[column] = table[column].map(partial(norm_min, column))

        elif normalisation_rule == 3:
            _min  = dict(table.copy().apply(lambda x: x.min()))
            _max  = dict(table.copy().apply(lambda x: x.max()))

            norm_min = lambda _key, element: _min[_key] / element
            norm_max = lambda _key, element: element / _max[_key]

            for column in table.columns:
                if direction[column]: # if True (==1), use the maximisation rule.
                    n_table[column] = table[column].map(partial(norm_max, column))

                else: # if False (==0), use the minimisation rule.
                    n_table[column] = table[column].map(partial(norm_min, column))

        elif normalisation_rule == 4:
            for column in table.columns:
                n_table[column] = preprocessing.scale(table[column])
            print(f'WARNING: normalisation rule {normalisation_rule} does not take into consideration MIN/MAX criteria')

        else:
            print('Please specify a normalisation rule i from the interval [1,4]')
            raise Exception('Invalid Normalization Rule')

        return n_table
    ##

    @staticmethod
    def _side_by_side_histograms(
        table1: pd.core.frame.DataFrame,
        table2: pd.core.frame.DataFrame,
        left_title: str = 'Before normalisation',
        right_title: str = 'After normalisation'
    ) -> NoReturn:
        """
        WARNING: This function assumes the two pandas.Dataframes have the same columns.
        """
        for i, column in enumerate(table1.columns):
            plt.figure(i)
            plt.subplot(1, 2, 1)
            sns.kdeplot(table1[column])
            plt.title(left_title)
            plt.subplot(1, 2, 2)
            sns.kdeplot(table2[column])
            plt.title(right_title)
        plt.show()
    ##

    @staticmethod
    def _concordance_matrix(
        frame: pd.core.frame.DataFrame,
        weights: Dict[str,float]
    ) -> pd.core.frame.DataFrame:
        """ Compute a concordance matrix from a normalized matrix (frame parameter)
            and a weights dictionary. Said weights dictionnary should map
            f:criterion -> weight, for all criteria contained in frame.columns
        """

        if not sorted(list(weights.keys())) == sorted(list(frame.columns)):
            e = f'weights paramter\'s keys {sorted(list(weights.keys()))} do not match the normalized matrix\'s criteria (columns) {sorted(list(frame.columns))}'
            raise Exception(e)

        _c_matrix = pd.DataFrame(columns=frame.index, index=frame.index)

        for option in frame.index:
            for option2 in frame.index:
                _sum = 0
                for criterion in frame.columns:
                    if frame.loc[option, criterion] > frame.loc[option2, criterion]:
                        _sum += weights[criterion]
                    elif np.isclose(frame.loc[option, criterion], frame.loc[option2, criterion]):
                        _sum += 0.5 * weights[criterion]
                if option == option2:
                    _c_matrix.loc[option, option2] = 0
                else:
                    _c_matrix.loc[option, option2] = _sum

        return _c_matrix
    ##

    @staticmethod
    def _discordance_matrix(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        """ Compute a concordance matrix from a weighted normalized matrix (frame parameter). """

        _d_matrix = pd.DataFrame(columns=frame.index, index=frame.index)

        negatives = lambda y: list(filter(lambda x: True if x < 0 else False, y))

        _discordance_index = 0
        for option in frame.index:
            for option2 in frame.index:
                diffs = list(frame.loc[option, :] - frame.loc[option2, :])
                if not any(diffs):
                    _discordance_index = 0
                else:
                    n_diffs = negatives(diffs)
                    if not n_diffs:
                        num = 0
                    else:
                        num = max(np.abs(negatives(diffs)))
                    denom = max(np.abs(diffs))
                    _discordance_index = num / denom
                _d_matrix.loc[option, option2] = _discordance_index

        return _d_matrix
    ##

    @staticmethod
    def _agregated_dominance_matrix(
        concordant_dominance_matrix: pd.core.frame.DataFrame,
        discordant_dominance_matrix: pd.core.frame.DataFrame
    ) -> pd.core.frame.DataFrame:
        """ Compute the agregated dominance matrix from the concordant
            dominance and the discordant dominance matrices.
        """

        if list(concordant_dominance_matrix.columns) != list(discordant_dominance_matrix.columns):
            raise Exception('Cannot compute ADM from matrices with different columns')
        if list(concordant_dominance_matrix.index) != list(discordant_dominance_matrix.index):
            raise Exception('Cannot compute ADM from matrices with different rows')

        _a_matrix = pd.DataFrame(
            columns=concordant_dominance_matrix.columns,
            index=concordant_dominance_matrix.index
        )

        for option in concordant_dominance_matrix.columns:
            for option2 in concordant_dominance_matrix.index:
                _a_matrix.loc[option, option2] = (lambda x, y: 1 if x and y else 0)(
                        concordant_dominance_matrix.loc[option, option2],
                        discordant_dominance_matrix.loc[option, option2]
                )
        return _a_matrix
    ##
