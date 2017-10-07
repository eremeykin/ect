class Tree:
    def __init__(self, root=None):
        self.root = root
        if self.root is not None:
            self.left_tree = Tree(None)
            self.right_tree = Tree(None)

    @staticmethod
    def add(element, tree):
        if tree.root is None:
            tree.root = element
            tree.left_tree = Tree(None)
            tree.right_tree = Tree(None)
        else:
            if tree.root.key > element.key:
                Tree.add(element, tree.right_tree)
            elif tree.root.key < element.key:
                Tree.add(element, tree.left_tree)
            else:
                tree.root.values.add(element.value)

    @staticmethod
    def remove(key, value, tree):
        if tree.root is None:
            return
        else:
            if key > tree.root.key:
                Tree.remove(key, tree.right_tree)
            elif key < tree.root.key:
                Tree.remove(key, tree.left_tree)
            else:
                if len(tree.root.values) > 1:
                    tree.root.values.remove(value)
                else:
                    if tree.left_tree.root is None and tree.right_tree.root is None:
                        tree.root = None
                    elif tree.left_tree.root is None:
                        tree.root = tree.right_tree.root
                        tree.left_tree = tree.right_tree.left_tree
                        tree.right_tree = tree.right_tree.right_tree
                    elif tree.right_tree.root is None:
                        tree.root = tree.left_tree.root
                        tree.right_tree = tree.left_tree.right_tree
                        tree.left_tree = tree.left_tree.left_tree
                    else:
                        if tree.right_tree.left_tree.root is None:
                            self.root =

    def show(self):
        if self.root is not None:
            print("([{}] (L {}) (R {}))".format(self.root.key,
                                                self.left_tree.root.key if self.left_tree.root is not None else None,
                                                self.right_tree.root.key if self.right_tree.root is not None else None))
            self.left_tree.show()
            self.right_tree.show()


class TreeElement:
    def __init__(self, key, value):
        self.key = key
        self.values = [value]

import numpy as np

if __name__ == "__main__":
    te = TreeElement(500, 5)
    tree = Tree(te)
    keys = np.unique(np.random.randint(0, 1000, 30))
    np.random.shuffle(keys)
    for x in keys:
        tree.add(TreeElement(x, 0), tree)
    tree.show()
    print(max(keys))
