# Standard IPython notebook imports

import random
import graphviz
import numpy as np
import pandas as pd
import scipy.stats as sstats

def EntropyRate(counts):
    sm = counts.sum()
    counts /= sm

    ans = 0
    for s in counts:
        if s > 0:
            ans += s / sm * np.log2(s / sm)
    return -ans

def GiniRate(counts):
    s = counts.sum()
    counts /= s
    return 1. - counts.dot(counts)


def MeanErrRate(counts):
    s = counts.sum()
    counts /= s
    return 1. - counts.max()

class AbstractSplit:
    """Split the examples in a tree node according to a criterion.
    """

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, x):
        """Return the subtree corresponding to x."""
        raise NotImplementedError

    def build_subtrees(self, df, subtree_kwargs):
        """Recuisively build the subtrees."""
        raise NotImplementedError

    def iter_subtrees(self):
        """Return an iterator over subtrees."""
        raise NotImplementedError

    def add_to_graphviz(self, dot):
        """Add the split to the graphviz vizalization."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}: {self.attr}"

class CategoricalMultivalueSplit(AbstractSplit):
    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        for group_name, group_df in df.groupby(self.attr):
            child = Tree(group_df, **subtree_kwargs)
            self.subtrees[group_name] = child

    def __call__(self, x):
        # Return the subtree for the given example
        if x[self.attr] in self.subtrees:
            return self.subtrees[x[self.attr]]
        else:
            return None

    def iter_subtrees(self):
        return self.subtrees.values()

    def add_to_graphviz(self, dot, parent, print_info):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{split_name}")

def GetCategoricalSplit(
    df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
):
    split = CategoricalMultivalueSplit(attr)

    mean_child_purity = 0
    for group_name, group_df in df.groupby(attr):
        mean_child_purity += purity_fun(group_df['target'].value_counts()) * group_df.shape[0]
    mean_child_purity /= df.shape[0]

    purity_gain = parent_purity - mean_child_purity
    if normalize_by_split_entropy:
        purity_gain /= EntropyRate(df[attr].value_counts())
    return split, purity_gain

class NumericalSplit(AbstractSplit):
    def __init__(self, attr, th):
        super(NumericalSplit, self).__init__(attr)
        self.th = th

    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = (
            Tree(df[df[self.attr] <= self.th], **subtree_kwargs),
            Tree(df[df[self.attr] > self.th], **subtree_kwargs),
        )

    def __call__(self, x):
        # return the sobtree for the data sample `x`
        if x[self.attr] <= self.th:
            return self.subtrees[0]
        return self.subtrees[1]

    def __str__(self):
        return f"NumericalSplit: {self.attr} <= {self.th}"

    def iter_subtrees(self):
        return self.subtrees

    def AddToGraphviz(self, dot, parent, print_info):
        self.subtrees[0].AddToGraphviz(dot, print_info)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[0])}", label=f"<= {self.th:.2f}")
        self.subtrees[1].AddToGraphviz(dot, print_info)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[1])}", label=f"> {self.th:.2f}")


def GetNumericalSplit(
    df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
):
    """Find best split thereshold and compute the average purity after a split.
    Args:
        df: a dataframe
        parent_purity: purity of the parent node
        purity_fun: function to compute the purity
        attr: attribute over whihc to split the dataframe
        normalize_by_split_entropy: if True, divide the purity gain by the split
            entropy (to compute https://en.wikipedia.org/wiki/Information_gain_ratio)
    
    Returns:
        pair of (split, purity_gain)
    """
    attr_df = df[[attr, "target"]].sort_values(attr)
    targets = attr_df["target"]
    values = attr_df[attr]
    # Start with a split that puts all the samples into the right subtree
    right_counts = targets.value_counts()
    left_counts = right_counts * 0

    best_split = None  # Will be None, or NumericalSplit(attr, best_threshold)
    best_purity_gain = -1
    N = len(attr_df)
    for row_i in range(N - 1):
        # Update the counts of targets in the left and right subtree and compute
        # the purity of the slipt for all possible thresholds!
        # Return the best split found.

        # Remember that the attribute may have duplicate values and all samples
        # with the same attribute value must end in the same subtree!
        row_target = targets.iloc[row_i]
        attribute_value = values.iloc[row_i]
        next_attribute_value = values.iloc[row_i + 1]
        split_threshold = (attribute_value + next_attribute_value) / 2.0

        # Consider the split at threshold, i.e. NumericalSplit(attr, split_threshold)

        # the loop should return the best possible split.

        # TODO: update left_counts and right_counts
        left_counts[row_target] += 1
        right_counts[row_target] -= 1

        # The split is possible if attribute_value != next_attribute_value
        if attribute_value == next_attribute_value:
            continue

        # TODO: now consider the split at split_threshold and save it if it the best one
        mean_child_purity = ((row_i + 1) * purity_fun(left_counts.copy()) + (N - row_i - 1) * purity_fun(right_counts.copy())) / N
        purity_gain = parent_purity - mean_child_purity

        if normalize_by_split_entropy:
            purity_gain /= EntropyRate(np.array([row_i + 1, N - row_i - 1]))

        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = NumericalSplit(attr, split_threshold)
    return best_split, best_purity_gain

def GetSplit(df, criterion="infogain", nattrs=None):
    """Find best split on the given dataframe.
    
    Attributes:
        - df: the dataframe of smaples in the node to be split
        - criterion: spluis selection criterion
        - nattrs: flag to randomly limit the number of considered attributes. Used 
          in random tree impementations.

    Returns:
        - If no split exists, return None.
        - If a split exists, return an instance of a subclass of AbstractSplit
    """

    target_value_counts = df["target"].value_counts()
    if len(target_value_counts) == 1:
        return None

    possible_splits = [s for s in df.columns if s != 'target' and df[s].nunique() > 1]  # possible_splits must be a list
    assert "target" not in possible_splits

    if not possible_splits:
        return None

    # Get the base purity measure and the purity function
    if criterion in ["infogain", "infogain_ratio"]:
        purity_fun = EntropyRate
    elif criterion in ["mean_err_rate"]:
        purity_fun = MeanErrRate
    elif criterion in ["gini"]:
        purity_fun = GiniRate
    else:
        raise Exception("Unknown criterion: " + criterion)
    base_purity = purity_fun(target_value_counts)

    best_purity_gain = -1
    best_split = None

    if nattrs is not None:
        possible_splits = [s for s in np.random.choice(possible_splits, nattrs)]
    
    for attr in possible_splits:
        if np.issubdtype(df[attr].dtype, np.number):
            split_sel_fun = GetNumericalSplit
        else:
            split_sel_fun = GetCategoricalSplit

        split, purity_gain = split_sel_fun(
            df,
            base_purity,
            purity_fun,
            attr,
            normalize_by_split_entropy=criterion.endswith("ratio"),
        )

        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = split
    return best_split

class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()
        assert not df.isnull().values.any()

        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())
        kwargs_orig = dict(kwargs)

        self.all_targets = kwargs.pop("all_targets")

        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": EntropyRate(self.counts),
            "gini": GiniRate(self.counts),
        }

        self.split = GetSplit(df, **kwargs)
        if self.split:
            self.split.build_subtrees(df, kwargs_orig)

    def GetTargetDistribution(self, sample):
        if self.split is None:
            return self.counts
        else:
            subtree = self.split(sample)
            if subtree == None:
                return self.counts
            return subtree.GetTargetDistribution(sample)
            
    def Classify(self, sample):
        result = self.GetTargetDistribution(sample)
        return result.index[0]
        
    def Draw(self, print_info=True):
        dot = graphviz.Digraph()
        self.AddToGraphviz(dot, print_info)
        return dot

    def AddToGraphviz(self, dot, print_info):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []

        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i%9 + 1};{freq}")
                freqs_info.append(f"{c}:{freq:.2f}")

        colors = ":".join(colors)
        labels = [" ".join(freqs_info)]

        if print_info:
            for k, v in self.info.items():
                labels.append(f"{k} = {v}")

        if self.split:
            labels.append(f"split by: {self.split.attr}")

        dot.node(
            f"{id(self)}",
            label="\n".join(labels),
            shape="box",
            style="striped",
            fillcolor=colors,
            colorscheme="set19",
        )

        if self.split:
            self.split.AddToGraphviz(dot, self, print_info)