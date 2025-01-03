from data_class import mydata
from modeles import *

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def remove_highly_correlated_features(X, threshold=0.8):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    print("Calculating correlation matrix...")
    corr_matrix = X.corr().abs()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)

    print(f"Dropping {len(to_drop)} highly correlated features.")
    return X.drop(columns=to_drop), list(X.columns.difference(to_drop))


def forward_selection_with_correlation_check(model, X_train, y_train, X_test, y_test, feature_names,
                                             correlation_threshold=0.8):
    print("Removing highly correlated features...")
    X_filtered, filtered_feature_names = remove_highly_correlated_features(X_train, correlation_threshold)

    selected_features = []
    remaining_features = list(filtered_feature_names)
    best_score = 0
    best_features = []

    print("Starting forward selection...")
    while remaining_features:
        scores = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            model.fit(X_filtered[current_features], y_train)
            y_pred = model.predict(X_filtered[current_features])
            report = classification_report(y_train, y_pred, output_dict=True, zero_division=0)
            score = report['weighted avg']['f1-score']
            scores.append((score, feature))

        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_feature = scores[0]

        print(f"Best feature this iteration: {best_feature} with F1-score: {best_score:.4f}")

        if best_score > 0:  # Если есть улучшение
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_features = selected_features.copy()  # Обновляем лучший набор признаков
        else:
            break  # Если нет улучшения, выходим из цикла

    # Проверка на тестовом наборе
    if best_features:  # Проверяем, что есть выбранные признаки
        print("Fitting model on the test set...")
        model.fit(X_filtered[best_features], y_train)
        X_test_filtered, _ = remove_highly_correlated_features(X_test, correlation_threshold)  # Применяем к тестовому набору
        y_test_pred = model.predict(X_test_filtered[best_features])  # Используем отфильтрованный тестовый набор
        test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        test_f1_score = test_report['weighted avg']['f1-score']
    else:
        test_f1_score = 0  # Если нет выбранных признаков, устанавливаем F1 в 0

    print("Forward selection completed.")
    return best_features, best_score, test_f1_score


def eval_futures_with_forward_selection(X_train, y_train, X_test, y_test, feature_names):
    model = LogisticRegression(max_iter=500, random_state=42)
    best_features, best_score, test_f1_score = forward_selection_with_correlation_check(model, X_train, y_train, X_test,
                                                                                        y_test, feature_names)
    # Преобразуем индексы в названия признаков
    best_feature_names = [feature_names[int(feature)] for feature in best_features]

    return best_feature_names, best_score, test_f1_score

md = mydata()
df = md.load_data_from_local('data/DOGE5.json')
data = md.take_data(df)
X_train, y_train, X_val, y_val, X_test, y_test, feature_names = split_scale_data(data)

best_feature_names, best_score, test_f1_score = eval_futures_with_forward_selection(X_train, y_train, X_test, y_test,
                                                                                     feature_names)
print("Лучшие признаки:", best_feature_names)
print("Лучший F1-score на обучающем наборе:", best_score)
print("F1-score на тестовом наборе:", test_f1_score)