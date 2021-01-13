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

# Just a class to inherit from
class AbstractSplit:
    def __init__(self, attr):
        self.attr = attr

    def __call__(self, x):
        raise NotImplementedError

    def buildSubtrees(self, df, subtreeKwargs):
        raise NotImplementedError

    def iterSubtrees(self):
        raise NotImplementedError

    def addToGraphviz(self, dot):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}: {self.attr}"

class CategoricalMultivalueSplit(AbstractSplit):
    def buildSubtrees(self, df, subtreeKwargs):
        self.subtrees = {}
        for groupName, groupDF in df.groupby(self.attr):
            child = Tree(groupDF, **subtreeKwargs)
            self.subtrees[groupName] = child

    def __call__(self, x):
        if x[self.attr] in self.subtrees:
            return self.subtrees[x[self.attr]]
        else:
            return None

    def iterSubtrees(self):
        return self.subtrees.values()

    def addToGraphviz(self, dot, parent, printInfo):
        for splitName, child in self.subtrees.items():
            child.addToGraphviz(dot, printInfo)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{splitName}")

def GetCategoricalSplit(df, parentPurity, purityFun, attr, normalizeBySplitEntropy=False):
    split = CategoricalMultivalueSplit(attr)

    meanChildPurity = 0
    for groupName, groupDF in df.groupby(attr):
        meanChildPurity += purityFun(groupDF['target'].value_counts()) * groupDF.shape[0]
    meanChildPurity /= df.shape[0]

    purityGain = parentPurity - meanChildPurity
    if normalizeBySplitEntropy:
        purityGain /= EntropyRate(df[attr].value_counts())
    return split, purityGain

class NumericalSplit(AbstractSplit):
    def __init__(self, attr, th):
        super(NumericalSplit, self).__init__(attr)
        self.th = th

    def buildSubtrees(self, df, subtreeKwargs):
        self.subtrees = (
            Tree(df[df[self.attr] <= self.th], **subtreeKwargs),
            Tree(df[df[self.attr] > self.th], **subtreeKwargs),
        )

    def __call__(self, x):
        # return the sobtree for the data sample `x`
        if x[self.attr] <= self.th:
            return self.subtrees[0]
        return self.subtrees[1]

    def __str__(self):
        return f"NumericalSplit: {self.attr} <= {self.th}"

    def iterSubtrees(self):
        return self.subtrees

    def AddToGraphviz(self, dot, parent, printInfo):
        self.subtrees[0].AddToGraphviz(dot, printInfo)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[0])}", label=f"<= {self.th:.2f}")
        self.subtrees[1].AddToGraphviz(dot, printInfo)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[1])}", label=f"> {self.th:.2f}")


def GetNumericalSplit( df, parentPurity, purityFun, attr, normalizeBySplitEntropy=False):
    attrDF = df[[attr, "target"]].sort_values(attr)
    targets = attrDF["target"]
    values = attrDF[attr]

    rightCounts = targets.value_counts()
    leftCounts = rightCounts * 0

    bestSplit = None
    bestPurityGain = -1
    N = len(attrDF)
    for rowI in range(N - 1):
        rowTarget = targets.iloc[rowI]
        attributeValue = values.iloc[rowI]
        nextAttributeValue = values.iloc[rowI + 1]
        splitThreshold = (attributeValue + nextAttributeValue) / 2.0

        leftCounts[rowTarget] += 1
        rightCounts[rowTarget] -= 1

        if attributeValue == nextAttributeValue:
            continue

        leftCountschildPurity = ((rowI + 1) * purityFun(leftCounts.copy())
        rightcountschildPurity (N - rowI - 1) * purityFun(rightCounts.copy()))
        meanChildPurity = (leftcountschildPurity + rightcountschildPurity) / N
        purityGain = parentPurity - meanChildPurity

        if normalizeBySplitEntropy:
            purityGain /= EntropyRate(np.array([rowI + 1, N - rowI - 1]))

        if purityGain > bestPurityGain:
            bestPurityGain = purityGain
            bestSplit = NumericalSplit(attr, splitThreshold)
    return bestSplit, bestPurityGain

def GetSplit(df, criterion="infogain", nattrs=None):
    targetValueCounts = df["target"].value_counts()
    if len(targetValueCounts) == 1:
        return None

    possibleSplits = [s for s in df.columns if s != 'target' and df[s].nunique() > 1]
    assert "target" not in possibleSplits

    if not possibleSplits:
        return None

    # Get the base purity measure and the purity function
    if criterion in ["infogain", "infogain_ratio"]:
        purityFun = EntropyRate
    elif criterion in ["mean_err_rate"]:
        purityFun = MeanErrRate
    elif criterion in ["gini"]:
        purityFun = GiniRate
    else:
        raise Exception("Unknown criterion: " + criterion)

    basePurity = purityFun(targetValueCounts)
    bestPurityGain = -1
    bestSplit = None

    if nattrs is not None:
        possibleSplits = [s for s in np.random.choice(possibleSplits, nattrs)]
    
    for attr in possibleSplits:
        if np.issubdtype(df[attr].dtype, np.number):
            splitSelFun = GetNumericalSplit
        else:
            splitSelFun = GetCategoricalSplit

        split, purityGain = splitSelFun(
            df,
            basePurity,
            purityFun,
            attr,
            normalizeBySplitEntropy = criterion.endswith("ratio"),
        )

        if purityGain > bestPurityGain:
            bestPurityGain = purityGain
            bestSplit = split
    return bestSplit

class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()
        assert not df.isnull().values.any()

        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())
        kwargsOrig = dict(kwargs)

        self.allTargets = kwargs.pop("all_targets")
        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": EntropyRate(self.counts),
            "gini": GiniRate(self.counts),
        }

        self.split = GetSplit(df, **kwargs)
        if self.split:
            self.split.buildSubtrees(df, kwargsOrig)

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
        if np.issubdtype(result.index.dtype, np.number):
            return np.array(result.index).dot(np.array(result.values)) / result.size
        return result.index[0]
        
    def Draw(self, printInfo=True):
        dot = graphviz.Digraph()
        self.AddToGraphviz(dot, printInfo)
        return dot

    def AddToGraphviz(self, dot, printInfo):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqsInfo = []

        for i, c in enumerate(self.allTargets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i%9 + 1};{freq}")
                freqsInfo.append(f"{c}:{freq:.2f}")

        colors = ":".join(colors)
        labels = [" ".join(freqsInfo)]

        if printInfo:
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
            self.split.AddToGraphviz(dot, self, printInfo)
