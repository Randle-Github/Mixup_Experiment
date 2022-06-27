import os
import json

from collections import namedtuple

ListData = namedtuple('ListData', ['id', 'label', 'path'])

#  NOTE: 这个类在 VideoFolder 中被调用，但似乎没有什么作用
class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, json_path_input, json_path_labels, data_root,
                 extension, is_test=False):
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict()
        self.json_data = self.read_json_input()

    def read_json_input(self):
        json_data = []
        if not self.is_test:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    label = self.clean_template(elem['template'])
                    if label not in self.classes:
                        raise ValueError("Label mismatch! Please correct")
                    item = ListData(elem['id'],
                                    label,
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        else:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data

    def read_json_labels(self):
        ''' classes is a list, classes[0]="Moving something up" for examples'''
        classes = []
        with open(self.json_path_labels, 'r') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    # NOTE: 这里的 classes_dict 顺序就按照 label.json 来
    # CAUTION!: 修改后，原先 pretrain 的模型就不能起作用了，因为 action 的索引发生了变化
    def get_two_way_dict(self):
        '''
        You can index a class name by its label or index a label by its name
        '''
        classes_dict = {}
        with open(self.json_path_labels, 'r') as jsonfile:
            json_reader = json.load(jsonfile)
            for label, idx in json_reader.items():
                classes_dict[label] = idx
                classes_dict[idx] = label
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template

class WebmDataset(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".webm"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)

