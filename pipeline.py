from preprocessing import pre_process
from classification import apply_training

def pipeline(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols, batch_size, epochs, augment):
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape = pre_process(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols)
    model, history = apply_training(x_train, y_train, x_val, y_val, batch_size, epochs, input_shape, num_classes, augment)
    return model, history, x_train, y_train, x_val, y_val, x_test, y_test, input_shape