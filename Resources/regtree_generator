import pandas as pd
import numpy as np

class Node:
    def __init__(self, train_datapoints: pd.DataFrame, target,
                 is_Terminal: bool = True, split: float = None, feature: str = None,
                 lt_node=None, gte_node=None,  
                 parent=None, depth: int = 0,
                 generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        self.train_datapoints = train_datapoints
        self.target = target
        self.mean = np.array(self.train_datapoints[target]).mean()
        self.err = np.sum((np.array(self.train_datapoints[target]) - self.mean)**2)
        self.is_Terminal = is_Terminal
        self.split = split
        self.feature = feature
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.parent = parent
        self.depth = depth
        self.generate = generate
        self.max_depth = max_depth
        self.min_points = min_points
        # This allows nodes to recursively generate children nodes until the max_depth or the min_points has been reached.
        if self.generate and self.depth <= self.max_depth and self.train_datapoints.shape[0] >= self.min_points:
            self.create_child(generate=self.generate, max_depth=self.max_depth, min_points=self.min_points)

    def add_Nodes(self, lt_node, gte_node):
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.lt_node.parent = self
        self.gte_node.parent = self
        self.is_Terminal = False

    def evaluate_point(self, features):
        if self.is_Terminal:
            # print(f"The value is: {self.mean}")
            return self.mean
        else:
            if features[self.feature] < self.split:
                return self.lt_node.evaluate_point(features)
            else:
                return self.gte_node.evaluate_point(features)

    def create_child(self, generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        # If generate = True, it will recursively create children nodes.
        # This method will find the best split of the datapoints on each node that minimizes the sse.
        def feature_best_split(X, y):
            # This function chooses the best split for the individual features.
            X_sorted, y_sorted = np.array(X), np.array(y)

            best_sse = float('inf')
            best_split = None
            for i in range(1, len(X_sorted)):
                s = (X_sorted[i-1] + X_sorted[i]) / 2  # Candidate split point
                left_mask = X_sorted <= s
                right_mask = X_sorted > s

                y_left, y_right = y_sorted[left_mask], y_sorted[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # Skip empty splits

                sse = np.sum((y_left - y_left.mean())**2) + np.sum((y_right - y_right.mean())**2)

                if sse < best_sse: # Getthing the best split that minimises the SSE for that particular feature
                    best_sse, best_split = sse, s
            return best_split, best_sse

        def best_split(df: pd.DataFrame, target):
            # This function chooses the best split among all features .
            best_sse = float('inf')
            best_s = None
            best_feature = None
            for feature in df.columns:
                if feature == target:
                    continue
                df_sorted = df.sort_values(by=feature)
                split, sse = feature_best_split(df_sorted[feature], df_sorted[target])
                if sse < best_sse: # Getthing the best split among all features that minimises the SSE
                    best_sse, best_s, best_feature = sse, split, feature
            return best_sse, best_s, best_feature

        _, best_s, best_feature = best_split(self.train_datapoints, self.target)

        # The algorithm below uses the defined functions above to apply binary splitting.

        if best_s is not None and best_feature is not None:
            self.split = best_s
            self.feature = best_feature

            lt_points = self.train_datapoints[self.train_datapoints[best_feature] < best_s]
            gte_points = self.train_datapoints[self.train_datapoints[best_feature] >= best_s]

            if len(lt_points) > self.min_points and len(gte_points) > self.min_points:
                lt_node = Node(train_datapoints=lt_points, target=self.target, depth=self.depth + 1,
                               generate=generate, max_depth=max_depth, min_points=min_points)
                gte_node = Node(train_datapoints=gte_points, target=self.target, depth=self.depth + 1,
                                generate=generate, max_depth=max_depth, min_points=min_points)
                self.add_Nodes(lt_node=lt_node, gte_node=gte_node)

    def generate_children(self, max_depth: float, min_points: int):
        # This method initiates the recursive binary splitting from the root node and recursively builds the nodes of the tree.
        # The stopping criteria is the max_depth (the maximum depth for each node) and the min_points (the minimum training datapoints a node can contain)
        if self.is_Terminal:
            self.create_child(generate=True, max_depth=max_depth, min_points=min_points)

class Tree:
    def __init__(self, root_node: Node = None, training_data: pd.DataFrame = None, target: str = None):
        self.root_node = root_node       
        self.training_data = training_data
        self.target = target
        self.features = None
        self.depth = None
        self.err = float('inf') # The training SSE of the tree
        self.residuals = None
        if self.target is not None:
            self.get_features()

    def load_data(self, training_data: pd.DataFrame, target: str):
        if self.training_data is None:
            self.training_data = training_data
            self.target = target
        else:
            print("Data has already been loaded!")

    def create_root_node(self):
        if self.root_node is None:
            if self.training_data is None:
                print("Load the data first!")
            else:
                self.root_node = Node(train_datapoints=self.training_data, target=self.target)
        else:
            
            print("The tree already has a root node!")

    def evaluate(self, feature_set: pd.DataFrame):
        if self.root_node is None:
            print("Generate a tree first!")
        else:
            predictions = []
            for _, features in feature_set.iterrows():
                predictions.append(self.root_node.evaluate_point(features))
            return np.array(predictions)
        
    def get_features(self):
        features = []
        for feature in self.training_data.columns:
            if feature == self.target:
                continue
            features.append(feature)
        self.features = features

    def calculate_error(self):
        self.err = np.sum((np.array(self.training_data[self.target]) - self.evaluate(self.training_data[self.features]))**2)

    def calculate_residuals(self):
        self.residuals = np.array(self.training_data[self.target]) - self.evaluate(self.training_data[self.features])

    def generate_tree(self, max_depth: float = float('inf'), min_points: int = 1):
        if self.root_node is None:
            self.create_root_node()
        self.root_node.generate_children(max_depth=max_depth, min_points=min_points)
        self.calculate_error()
        self.calculate_residuals()

