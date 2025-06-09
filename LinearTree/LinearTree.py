import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from Util.Transformation import ConditionalTransformation, Partition, SingleTransformation, Transformation
from Util.Util import is_categorical, maybe_primary_key, RELATIONAL_OPERATORS, flip
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations
import os

EPS = 1.0               # Very small loss and if loss is this small, can assume perfect fit
MIN_PARTITION_SIZE = 2  # minimum size of a valid partition

class LinearTree:
    def __init__(self, 
                 source_df: pd.DataFrame, 
                 target_df: pd.DataFrame, 
                 target_col = None, 
                 potential_split_attributes = None,
                 potential_transformation_attributes = None,                  
                 max_depth = 5,
                 conditions = []):
        self.source_df = source_df
        self.target_df = target_df
        self.target_col = target_col
        if self.target_col is None:
            changed_cols = [col for col in source_df.columns if not np.array_equal(np.array(source_df[col]), np.array(target_df[col]))]
            assert len(changed_cols) == 1, "Currently change in exactly one target attribute is supported"
            self.target_col = changed_cols[0]  

        self.potential_split_attributes = potential_split_attributes
        if self.potential_split_attributes is None:
            # TODO: For now, split attribute is limited to ONLY categorical attributes, but this can be extended
            self.potential_split_attributes = [attr for attr in self.source_df.columns if is_categorical(self.source_df[attr])]

        self.potential_transformation_attributes = potential_transformation_attributes
        if self.potential_transformation_attributes is None:
            self.potential_transformation_attributes = [attr for attr in self.source_df.columns if 
                                                    not is_categorical(self.source_df[attr]) and
                                                    not maybe_primary_key(self.source_df[attr]) and
                                                    is_numeric_dtype(self.source_df[attr]) and 
                                                    attr != self.target_col]        
        
        self.max_depth = max_depth
        self.conditions = conditions            # conditions passed from parent, for root node, it is empty        

        self.split_condition = None             # how to best split this node
        self.left_tree = None                   # Unless leaf, will be none
        self.right_tree = None                  # Unless leaf, will be none
        self.model = None                       # Unless leaf, will be none
        self.transformation_attributes = None   # Unless leaf, will be none
        self.loss = 1e100
        self.transformation = None
        self.fit()


        merge_happened = True

        while merge_happened:
            merge_happened = False
            new_cts = []
            invalids = []

            for i in range(len(self.transformation.conditional_transformations) - 1):
                for j in range(i + 1, len(self.transformation.conditional_transformations)):
                    ct1 = self.transformation.conditional_transformations[i]
                    ct2 = self.transformation.conditional_transformations[j]
                    
                    if ct1.single_transformation.matches(ct2.single_transformation):
                        merged_partition = ct1.partition.merge(ct2.partition)
                        if merged_partition is not None:
                            merged_partition.process(self.source_df)
                            merged_ct = ConditionalTransformation(
                                 partition=merged_partition, 
                                 single_transformation=ct1.single_transformation)
                            new_cts.append(merged_ct)
                            invalids.append(i)
                            invalids.append(j)
                            merge_happened = True
            
            for i in range(len(self.transformation.conditional_transformations)):
                if i not in invalids:
                     new_cts.append(self.transformation.conditional_transformations[i])
            
            self.transformation = Transformation(conditional_transformations=new_cts)
            
                      
             

    def fit(self):
        if np.array_equal(self.target_df[self.target_col], self.source_df[self.target_col]):
            # No change happened, no need to explore this node
            self.model = None
            self.loss = 0
            self.losses = [self.loss]
            self.transformation = Transformation(conditional_transformations=[])
            return
        
        possible_transformation_attribute_set = []
        for n in range (1, len(self.potential_transformation_attributes) + 1):
            possible_transformation_attribute_set += [list(a) for a in list(map(set, combinations(self.potential_transformation_attributes, n)))]

        for trans_attributes in possible_transformation_attribute_set:
            model = LinearRegression()
            model.fit(X = self.source_df[trans_attributes], y = self.target_df[self.target_col])
            predicted_values = model.predict(X = self.source_df[trans_attributes])
            cur_loss = mean_squared_error(y_true = self.target_df[self.target_col], y_pred = predicted_values)
            if cur_loss < self.loss:
                self.loss = cur_loss
                self.model = model
                self.transformation_attributes = trans_attributes

        if self.max_depth == 0 or self.loss < EPS:     # no more splitting is allowed, just try to find the best fitting LR line OR loss is already very low                
            cur_partition = Partition(conditions = self.conditions)
            cur_partition.process(self.source_df)
            self.transformation = Transformation(conditional_transformations=[ConditionalTransformation(
                partition = cur_partition,
                single_transformation = SingleTransformation(
                     target_attribute = self.target_col,
                     independent_attributes = self.transformation_attributes,
                     coefficients = list(self.model.coef_) + [self.model.intercept_]))])
            self.losses = [self.loss]
            return

        # Try to find split to improve loss. Pick greedily.
        min_gain = 0
        best_split_parameters = None
        for attr in self.potential_split_attributes:
            potential_operators = []
            if is_numeric_dtype(self.source_df[attr]):
                    potential_operators = RELATIONAL_OPERATORS
            else:
                    potential_operators = RELATIONAL_OPERATORS[:2]
            for op in potential_operators:
                for val in np.unique(self.source_df[attr]):                
                    left_partition = Partition(conditions = [(attr, op, val)])
                    relevant_source_df = left_partition.apply(self.source_df)
                    relevant_target_df = left_partition.apply(self.target_df)
                    if relevant_source_df.shape[0] < MIN_PARTITION_SIZE:
                            continue
                    model.fit(X = relevant_source_df[self.transformation_attributes], y = relevant_target_df[self.target_col])
                    predicted_values = model.predict(X = relevant_source_df[self.transformation_attributes])
                    total_loss_at_left_partition = mean_squared_error(y_true = relevant_target_df[self.target_col], y_pred = predicted_values) * relevant_source_df.shape[0]

                    right_partition = Partition(conditions = [flip((attr, op, val))])
                    relevant_source_df = right_partition.apply(self.source_df)
                    if relevant_source_df.shape[0] < MIN_PARTITION_SIZE:
                            continue
                    relevant_target_df = right_partition.apply(self.target_df)
                    model.fit(X = relevant_source_df[self.transformation_attributes], y = relevant_target_df[self.target_col])
                    predicted_values = model.predict(X = relevant_source_df[self.transformation_attributes])
                    total_loss_at_right_partition = mean_squared_error(y_true = relevant_target_df[self.target_col], y_pred = predicted_values) * relevant_source_df.shape[0]

                    gain = self.loss * self.source_df.shape[0] - (total_loss_at_left_partition + total_loss_at_right_partition)
                    if gain > min_gain:
                        min_gain = gain
                        best_split_parameters = (attr, op, val)

        if best_split_parameters is not None:
            cur_left_tree = LinearTree(source_df = Partition(conditions = [best_split_parameters]).apply(self.source_df),
                                        target_df = Partition(conditions = [best_split_parameters]).apply(self.target_df),
                                        potential_split_attributes = self.potential_split_attributes,
                                        potential_transformation_attributes = self.potential_transformation_attributes,
                                        target_col = self.target_col, 
                                        max_depth = self.max_depth - 1, 
                                        conditions  = self.conditions + [best_split_parameters])
            cur_right_tree = LinearTree(source_df = Partition(conditions = [flip(best_split_parameters)]).apply(self.source_df),
                                        target_df = Partition(conditions = [flip(best_split_parameters)]).apply(self.target_df),
                                        potential_split_attributes = self.potential_split_attributes,
                                        potential_transformation_attributes = self.potential_transformation_attributes,
                                        target_col = self.target_col, 
                                        max_depth = self.max_depth - 1, 
                                        conditions  = self.conditions + [flip(best_split_parameters)])
                
            if cur_left_tree.loss * cur_left_tree.source_df.shape[0] + cur_right_tree.loss * cur_right_tree.source_df.shape[0] < self.loss * self.source_df.shape[0]:
                self.loss = (cur_left_tree.loss * cur_left_tree.source_df.shape[0] + cur_right_tree.loss * cur_right_tree.source_df.shape[0])/self.source_df.shape[0]
                self.left_tree = cur_left_tree
                self.right_tree = cur_right_tree
                self.split_condition = best_split_parameters
                self.model = None
                self.transformation_attributes = None
                self.transformation = Transformation(conditional_transformations= cur_left_tree.transformation.conditional_transformations + cur_right_tree.transformation.conditional_transformations)
                self.losses = cur_left_tree.losses + cur_right_tree.losses
        
        if self.model is not None:  # It is best as a leaf
            cur_partition = Partition(conditions = self.conditions)
            cur_partition.process(self.source_df)
            self.transformation = Transformation(conditional_transformations=[ConditionalTransformation(
                partition=cur_partition,
                single_transformation = SingleTransformation(
                     target_attribute = self.target_col,
                     independent_attributes = self.transformation_attributes,
                     coefficients = list(self.model.coef_) + [self.model.intercept_]))])
            self.losses = [self.loss]
    
    def store(self, result_path_root, file_name):
        self.transformation.store(
             df = self.source_df,
             detailed_text_path=os.path.join(result_path_root, file_name),
             pickle_dump_path=os.path.join(result_path_root, file_name.replace('txt', 'pkl')))






    
