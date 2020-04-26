from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, data, labels, x_train=[], y_train=[], x_val=[], y_val=[], x_test=[], y_test=[], num_classes=[], img_rows=[], img_cols=[],
                 input_shape=[], train_size=0.7, val_size=0.15, test_size=0.15):
        self.data = data
        self.labels = labels
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.input_shape = input_shape
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        return

    def train_val_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=(self.val_size + self.test_size))
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_test, self.y_test,
                                                                  test_size=self.test_size / (1 - self.train_size))
        return





