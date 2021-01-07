# Usage

## Decision Tree

To use a decision tree, import the class `Tree` from DecisionTree.py.
To create a tree, call `Tree(data_frame)`:
* Optional argument 'nattrs'; allows picking from fewer attributes on each step of creating a tree

Soon, it will also support weighted samples for boosting and maximum tree depth.

After creation, a tree provides the following functionality:
* `Classify(sample)` -- classifies the sample; sample should be the same type as the objects given in the data frame
* `GetTargetDistribution(sample)` -- returns the distribution of the node where sample is classified.
* `Draw()` -- draws the decision tree, optional argument `printInfo` (`True` by default), which prints additional info about this tree.

## Random Forest

To use it, import the class `Forest` from RandomForest.py.

To create a forest just call ```Forest(data_frame, number_of_trees, attributes_to_pick, target_type)```:
* `number_of_trees` describes the expected number of trees in the created forest
* `attributes_to_pick` describes the number of attributes from which a decision tree should choose at each step
* `target_type` is either numerical or categorical

After creation, a tree provides the following functionality:
* `Query(sample)` -- classifies the sample; sample should be the same type as the objects given in the data frame. 
Optional argument `pref` allows to query only prefix `pref` of built trees.
