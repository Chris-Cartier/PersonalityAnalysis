# coding: utf-8

import csv
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np


def metrics_result(actual, predict):
    precision = round(metrics.precision_score(actual, predict, average='weighted'), 3)  # round() 第二个参数为精确小数点后几位
    recall = round(metrics.recall_score(actual, predict, average='weighted'), 3)
    f1 = round(metrics.f1_score(actual, predict, average='weighted'), 3)
    return precision, recall, f1


def liwc_svm(label_file_name, feature_file_name):

    label_file = open(label_file_name, 'r')
    csv_file1 = csv.reader(label_file)

    feature_file = open(feature_file_name, 'r')
    csv_file2 = csv.reader(feature_file)

    feature_list = []
    index = 0
    for line in csv_file2:
        if (index % 2) == 0:
            # print("{0} 是偶数".format(index))
            feature = line[0].split('\t')
            del feature[0]  # 第一个元素为空，去掉
            # print feature
            feature_list.append(feature)
            index += 1
        else:
            # print("{0} 是奇数".format(index))
            index += 1
            continue
    del feature_list[0]  # 第一个元素列表为['WC', 'Analytic', 'Clout'……]，去掉
    # print len(feature_list)
    feature_array = np.array(feature_list)
    print feature_array.shape

    label_list = []
    for line in csv_file1:
        authid = line[0]
        content = line[1]
        sex = line[2]
        label_value = [line[3], line[4], line[5], line[6]]  # 四个标签
        # print label_value
        label_list.append(label_value)

    label_array = np.array(label_list)
    print label_array.shape
    # print type(label_array[:, 1])

    label_name = ['e_i', 's_n', 't_f', 'j_p']
    kf = KFold(n_splits=10, shuffle=True)
    for number in range(len(label_name)):  # 针对每一个标签
        # print 'label:', label_name[number]
        ave_precision = 0
        ave_recall = 0
        ave_f1 = 0
        one_label_array = label_array[:, number]
        # print one_label_array.shape

        for train_index, test_index in kf.split(feature_array):  # 10-fold
            # print 'done'
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = feature_array[train_index], feature_array[test_index]
            # print X_train.shape, X_test.shape
            y_train, y_test = one_label_array[train_index], one_label_array[test_index]
            # print y_train.shape, y_test.shape

            # print 'SVC begins'   # SVC模型
            clf = SVC(kernel="rbf")
            # print 'SVC-fit begins'
            clf.fit(X_train, y_train)
            # print 'SVC-predict begins'
            y_predict = clf.predict(X_test)


            precision, recall, f1 = metrics_result(y_test, y_predict)
            # print "precision: ", precision
            # print "recall: ", recall
            # print "f1: ", f1
            ave_precision += precision
            ave_recall += recall
            ave_f1 += f1

        print
        print label_name[number]
        print "ave_precision: ", ave_precision/10
        print "ave_recall: ", ave_recall/10
        print "ave_f1: ", ave_f1/10
        print


if __name__ == "__main__":
    label_file_name = 'newdata/50g.csv'
    feature_file_name = 'liwc_results/50g.txt'
    liwc_svm(label_file_name, feature_file_name)






