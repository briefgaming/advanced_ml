from collections import defaultdict


class RadixNode:

    def __init__(self, label, is_end_of_word = False):
        self.children = defaultdict()
        self.label = label
        self.is_end_of_word = is_end_of_word


