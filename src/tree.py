
# TREE CLASS DECLARATION


class Node:
    def __init__(self, index, val, groups):
        self.left = None
        self.right = None
        self.groups = groups
        self.index = index
        self.val = val


class Tree:
    def __init__(self,root):
        self.root = root

