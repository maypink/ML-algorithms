import pandas as pd
import numpy as np
from typing import Dict, List
from funcs import info_gain, entropy


class DecisionTree:
    def __init__(self, df: pd.DataFrame, max_depth: int = float('inf'), min_leaf_size: int = float('-inf'),
                 max_leaves_amount: int = float('inf')):
        self.data = df
        self.tree = []
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_leaves_amount = max_leaves_amount

    def check_min_leaf_size(sample1: pd.DataFrame, sample2: pd.DataFrame):
        if sample1.shape[0] > self.min_leaf_size | sample2.shape[0] > self.min_leaf_size:
            return False

    def check_restrictions(self):
        leaves = 0
        for node in self.tree:
            if len(node['parents']) >= self.max_depth:  # check max depth
                return False
            if node['edges'] == None:
                leaves += 1

        if leaves >= self.max_leaves_amount:  # check max leaves amount
            return False
        return True

    def build_tree(self):
        sample1, sample2, split_feature, best_t = self.find_best_split(self.data)
        self.add_node(sample1, ' [< ', split_feature, best_t, None, ['root'])
        self.add_node(sample2, ' [>= ', split_feature, best_t, None, ['root'])

        for node in self.tree:
            if self.check_restrictions() == False:
                break
            if (node['edges'] == None) & (entropy(node['df']['species']) != 0):
                sample1, sample2, split_feature, best_t = self.find_best_split(node['df'])
                if self.check_min_leaf_size == False:
                    node['edges'] = False
                    break
                self.add_node(sample1, ' [< ', split_feature, best_t, None, node['parents'] + [node['name']])
                self.add_node(sample2, ' [>= ', split_feature, best_t, None, node['parents'] + [node['name']])
                node['edges'] = [split_feature + ' [< ' + str(best_t), split_feature + ' [>= ' + str(best_t)]
        return self.tree

    def add_node(self, sample: pd.DataFrame, sign: str, split_feature: str, best_t: float, edges: List[str] = None,
                 parents: List[str] = None):
        node = {'name': split_feature + sign + str(best_t), 'df': sample, 'edges': edges, 'parents': parents}
        self.tree.append(node)

    def find_best_split(self, df: pd.DataFrame):
        best_IG = float('-inf')
        best_t = 0
        split_feature = ''
        for feature in df.columns.values[:-1]:
            cur_t, cur_IG = self.find_best_split_on_feature(feature, df)
            if cur_IG > best_IG:
                best_t, best_IG, split_feature = cur_t, cur_IG, feature

        sample1 = df[df[split_feature] < best_t]
        sample2 = df[df[split_feature] >= best_t]
        return sample1, sample2, split_feature, best_t

    def find_best_split_on_feature(self, feature, df):
        if df[feature].empty:
            return 0, float('-inf')
        grid = np.linspace(min(df[feature].to_list()), max(df[feature].to_list()), 10)
        best_t = grid[0]
        best_IG = -1
        for t in grid:
            sample1 = df[df[feature] < t]
            sample2 = df[df[feature] >= t]
            IG = info_gain(df['species'], sample1['species'], (sample2['species']))
            if IG > best_IG:
                best_t, best_IG = t, IG
        return best_t, best_IG

    def get_params(self, node: Dict):
        feature = node['name'].split('[')[0]
        sign = node['name'].split('[')[1].split(' ')[0]
        t = node['name'].split('[')[1].split(' ')[1]
        return feature[:-1], sign, float(t)

    def check_on_edges(self, sample: pd.DataFrame, edges: List[str]):
        for edge in edges:
            for node in self.tree:
                if node['name'] == edge:
                    feature, sign, t = self.get_params(node)
                    if sign == '<':
                        if (sample[feature] < t) & (node['edges'] != None):
                            return self.check_on_edges(sample, node['edges'])
                        elif node['edges'] == None:
                            return node['df'].groupby('species').count().reset_index().sort_values(by=feature,
                                                                                                   ascending=False).iloc[
                                0][0]
                        else:
                            continue
                    else:
                        if sign == '>=':
                            if (sample[feature] >= t) & (node['edges'] != None):
                                return self.check_on_edges(sample, node['edges'])
                            elif node['edges'] == None:
                                return node['df'].groupby('species').count().reset_index().sort_values(by=feature,
                                                                                                       ascending=False).iloc[
                                    0][0]
                            else:
                                continue

    def predict(self, samples: pd.DataFrame):
        res = []
        for sample in samples.iterrows():
            sample = sample[1]
            for node in self.tree:
                feature, sign, t = self.get_params(node)
                if sign == '<':
                    if (sample[feature] < t) & (node['edges'] != None):
                        res.append(self.check_on_edges(sample, node['edges']))
                        break
                    elif (sample[feature] < t) & (node['edges'] == None):
                        res.append(node['df'].groupby('species').count().reset_index().sort_values(by=feature,
                                                                                                   ascending=False).iloc[
                                       0][0])
                        break
                    else:
                        continue
                else:
                    if sign == '>=':
                        if (sample[feature] >= t) & (node['edges'] != None):
                            res.append(self.check_on_edges(sample, node['edges']))
                            break
                        elif (sample[feature] >= t) & (node['edges'] == None):
                            res.append(node['df'].groupby('species').count().reset_index().sort_values(by=feature,
                                                                                                       ascending=False).iloc[
                                           0][0])
                            break
                        else:
                            continue
        return res