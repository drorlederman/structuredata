from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def run_benckmarks(x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test):
    # benchmarks
    y_pred_val, y_pred_test,_ = apply_classifier(1, x_train_vect, y_train, x_val_vect, y_val, x_test_vect, y_test)
    print("Random forest:")
    print("Validation accuracy:",metrics.accuracy_score(y_val, y_pred_val))
    print("Test set accuracy:",metrics.accuracy_score(y_test, y_pred_test))

    y_pred_val, y_pred_test,_ = apply_classifier(2, x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test)
    print("Gradient Boosting:")
    print("Validation accuracy:",metrics.accuracy_score(y_val, y_pred_val))
    print("Test set accuracy:",metrics.accuracy_score(y_test, y_pred_test))

    y_pred_val, y_pred_test,clf = apply_classifier(3, x_train_vect, y_train, x_val_vect, y_val, x_test_vect, y_test)
    print("Neural Networks:")
    print("Validation accuracy:", metrics.accuracy_score(y_val, y_pred_val))
    print("Test set accuracy:", metrics.accuracy_score(y_test, y_pred_test))
    import matplotlib.pyplot as plt
    loss_values = clf.estimator.loss_curve_
    plt.plot(loss_values)
    plt.show()

    return

def run_benckmarks2(train_set, valid_set, test_set):
    # benchmarks

    y_pred_val, y_pred_test = apply_classifier(1, x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test)
    print("Random forest:")
    print("Validation accuracy:",metrics.accuracy_score(y_val, y_pred_val))
    print("Test set accuracy:",metrics.accuracy_score(y_test, y_pred_test))

    y_pred_val, y_pred_test = apply_classifier(2, x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test)
    print("Gradient Boosting:")
    print("Validation accuracy:",metrics.accuracy_score(y_val, y_pred_val))
    print("Test set accuracy:",metrics.accuracy_score(y_test, y_pred_test))
    return

def apply_classifier(classifier_type, x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test):
    if (classifier_type == 1):
        clf = RandomForestClassifier(n_estimators=1000)
    elif (classifier_type == 2):
        clf = GradientBoostingClassifier(max_depth=2, n_estimators=1000, learning_rate=1.0)
    elif (classifier_type == 3):
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)


    clf.fit(x_train_vect,y_train)
    y_pred_val = clf.predict(x_val_vect)
    y_pred_test = clf.predict(x_test_vect)


    return y_pred_val, y_pred_test, clf

