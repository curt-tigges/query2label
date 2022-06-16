
class COCOCategorizer():
    def __init__(self):
        f = open('data/coco-labels-2014-2017.txt')
        category_list = [line.rstrip('\n') for line in f]
        self.cat_dict = {cat:key for cat, key in enumerate(category_list)}

    def get_labels(self, pred_list):
        labels = [self.cat_dict[i] for i in range(len(pred_list)) if pred_list[i]==1]
        return labels