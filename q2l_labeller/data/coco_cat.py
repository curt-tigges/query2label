class COCOCategorizer:
    """Creates list of English-language labels corresponding to COCO label vector

    Attributes:
        cat_dict (dict): Dictionary mapping label codes to names.
    """

    def __init__(self):
        """Creates label code-name mapping"""
        f = open("data/coco-labels-2014-2017.txt")
        category_list = [line.rstrip("\n") for line in f]
        self.cat_dict = {cat: key for cat, key in enumerate(category_list)}

    def get_labels(self, pred_list):
        """_summary_

        Args:
            pred_list (list of ints): Multi-hot list of label codes from prediction

        Returns:
            list of strings: List of label names in English.
        """
        labels = [self.cat_dict[i] for i in range(len(pred_list)) if pred_list[i] == 1]
        return labels
