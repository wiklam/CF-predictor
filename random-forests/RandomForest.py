import random
import pandas as pd
from collections import Counter
from DecisionTree import Tree

class Forest:
    def __init__(self, df, tree_count, picks, target, **kwargs):
        self.tree_count = tree_count
        self.trees = []
        self.target = target

        verbose = 0
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]

        n = df.shape[0]
        for i in range(self.tree_count):
            sample_ids = [s for s in range(n)]
            sample_ids = random.choices(sample_ids, k = n)
            sample_df = pd.DataFrame([df.iloc[s] for s in sample_ids])

            if verbose == 1:
                print('prepared to build %d' % i)
            self.trees.append(Tree(sample_df, nattrs = picks))

            if verbose == 1:
                print('done %d' % i)

    def QueryPref(self, query_obj, pref):
        results = []
        for i in range(pref):
            results.append(self.trees[i].Classify(query_obj))
        
        if self.target == "categorical":
            return Counter(results).most_common(1)[0][0]
        return sum(results) / len(results)

    def Query(self, query_obj):
        return self.QueryPref(query_obj, self.tree_count)