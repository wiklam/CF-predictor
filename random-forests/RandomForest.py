import random
import pandas as pd
from collections import Counter
from DecisionTree import Tree

class Forest:
    def __init__(self, df, treeCount, picks, targetType, **kwargs):
        self.treeCount = treeCount
        self.trees = []
        self.targetType = targetType

        verbose = 0
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        
        n = df.shape[0]
        for i in range(self.treeCount):
            sampleIds = [s for s in range(n)]
            sampleIds = random.choices(sampleIds, k = n)
            sampleDF = pd.DataFrame([df.iloc[s] for s in sampleIds])

            if verbose == 1:
                print('prepared to build %d' % i)
            self.trees.append(Tree(sampleDF, nattrs = picks))

            if verbose == 1:
                print('done %d' % i)

    def Query(self, queryObj, **kwargs):
        if "pref" not in kwargs:
            kwargs["pref"] = self.treeCount
        pref = kwargs["pref"]
        
        results = []
        for i in range(pref):
            results.append(self.trees[i].Classify(queryObj))
        
        if self.targetType == "categorical":
            return Counter(results).most_common(1)[0][0]
        return sum(results) / len(results)
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.treeCount}, {self.targetType}"
