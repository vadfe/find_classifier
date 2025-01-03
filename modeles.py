from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression,Perceptron,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
from joblib import Parallel, delayed

def split_scale_data(df):
    _res = df.copy()
    drop_columns = ['p_close', 'p_30m_close', 'p_1h_close', 'p_open', 'p_30m_open', 'p_1h_open',
                    'p_high', 'p_30m_high', 'p_1h_high', 'p_low', 'p_30m_low', 'p_1h_low',
                    'close', 'open', 'high', 'low', 'volume', 'ma_prs', 'ma_vol', 'target']
    #Удаление    коррелирующих    признаков
    threshold = 1.9  # Порог корреляции
    corr_matrix = df.corr().abs()
    # Получаем индексы признаков, которые нужно удалить
    #to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # Если корреляция выше порога
                colname = corr_matrix.columns[i]  # Получаем имя столбца
                drop_columns.append(colname)

    # Удаляем коррелирующие признаки из DataFrame
    #_res = _res.drop(columns=to_drop)
    test_size = int(0.05 * len(_res))  # Определяем размер тестовой выборки
    test_df = _res.iloc[-test_size:]  # Тестовая выборка - последние test_size наблюдений
    y_test = (test_df['target'] > 0).astype(int)
    test_x = test_df.drop(columns=drop_columns)
    feature_names = test_x.columns.tolist()
    train_val_df = _res.iloc[:-test_size]  # Оставшиеся данные для тренировочной и валидационной выборок
    train_val_df_shuffled = train_val_df.sample(frac=1, random_state=42).reset_index(
        drop=True)  # Перемешиваем оставшиеся данные
    train_val_y = (train_val_df_shuffled['target'] > 0).astype(int)  # Определяем X и y для перемешанных данных
    train_val_x = train_val_df_shuffled.drop(columns=drop_columns)

    # Разделяем перемешанные данные на тренировочную и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        train_val_x, train_val_y, test_size=0.2, random_state=42, stratify=train_val_y
    )

    # Нормализация данных
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Обучение скейлера только на тренировочных данных
    X_val_scaled = scaler.transform(X_val)  # Применение скейлера к валидационным данным
    X_test_scaled = scaler.transform(test_x)  # Применение скейлера к тестовым данным
    print("X_train_scaled ",X_train_scaled.shape,"y_train ", y_train.shape)
    print("X_val_scaled ", X_val_scaled.shape,"y_val ", y_val.shape)
    print("X_test_scaled ", X_test_scaled.shape,"y_test ", y_test.shape)
    print("Class distribution in train set:\n", y_train.value_counts())
    print("Class distribution in validation set:\n", y_val.value_counts())
    print("Class distribution in test set:\n", y_test.value_counts())
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_names


def train_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{type(clf).__name__}: fit start")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\n{type(clf).__name__}: fit end. training_time: {training_time}")

    y_pred_val = clf.predict(X_val)
    report_val = classification_report(y_val, y_pred_val, output_dict=True)

    y_pred_test = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    newrow = {
        'Classifier': type(clf).__name__,
        'Training Time (s)': training_time,
        'Val Precision': report_val['weighted avg']['precision'],
        'Val Recall': report_val['weighted avg']['recall'],
        'Val F1-score': report_val['weighted avg']['f1-score'],
        'Test Precision': report_test['weighted avg']['precision'],
        'Test Recall': report_test['weighted avg']['recall'],
        'Test F1-score': report_test['weighted avg']['f1-score'],
    }
    print(f"End train {newrow}")
    return newrow


def evaluate_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    selector = RFECV(model, step=1, cv=5)# Создание экземпляра RFECV
    selector.fit(X_train, y_train) # Обучение модели
    X_selected = selector.transform(X_train)# Получение выбранных признаков
    X_val_selected = selector.transform(X_val)# Преобразование валидационного и тестового наборов данных
    X_test_selected = selector.transform(X_test)
    # Сохранение списка выбранных признаков
    selected_features = [feature for feature, selected in zip(feature_names, selector.support_) if selected]
    result = train_classifier(selector, X_selected, y_train, X_val_selected, y_val, X_test_selected, y_test)# Обучение классификатора на выбранных признаках
    return {
        'name': type(model).__name__,
        'n_features': selector.n_features_,
        'Classifier': result['Classifier'],
        'Training Time (s)': result['Training Time (s)'],
        'Val Precision': result['Val Precision'],
        'Val Recall': result['Val Recall'],
        'Val F1-score': result['Val F1-score'],
        'Test Precision': result['Test Precision'],
        'Test Recall': result['est Recall'],
        'Test F1-score': result['Test F1-score'],
        'Selected Features': selected_features
    }


def eval_futures(_df):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = split_scale_data(_df)
    # Список классификаторов для перебора
    classifiers = [
        LogisticRegression(max_iter=500, random_state=42),
        Perceptron(max_iter=1000, random_state=42),
        XGBClassifier(eval_metric='logloss', random_state=42),
        CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=False),
        LGBMClassifier(random_state=42),
        HistGradientBoostingClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        GaussianNB(),
        BernoulliNB(),
        KNeighborsClassifier(n_neighbors=5),
        MLPClassifier(max_iter=500, random_state=42),
        MultinomialNB(),
        RandomForestClassifier(random_state=42),
        LinearSVC(random_state=42),
        BaggingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        ExtraTreesClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),

        QuadraticDiscriminantAnalysis(reg_param=0.1),
        LinearDiscriminantAnalysis(),
        VotingClassifier(estimators=[('logistic', LogisticRegression(max_iter=500, random_state=42)),
                                     ('mlp', MLPClassifier(max_iter=500, random_state=42)),
                                     ('hist_gb', HistGradientBoostingClassifier(random_state=42)),
                                     ('gaussian_nb', GaussianNB())], voting='hard')
    ]

    # Параллельное выполнение
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_classifier)(clf, X_train, y_train, X_val, y_val, X_test, y_test, feature_names) for clf in classifiers)

    results_df = pd.DataFrame(results)
    results_df.to_csv('classification_results.csv', index=False)
    print(f"Results saved to {'classification_results.csv'}")

    return results_df  # Возвращаем DataFrame с результатами

def eval_3(_df):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = split_scale_data(_df)

    classifiers = [
        LogisticRegression(max_iter=500, random_state=42),
        Perceptron(max_iter=1000, random_state=42),
        XGBClassifier(eval_metric='logloss', random_state=42),
        CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=False),
        LGBMClassifier(random_state=42),
        HistGradientBoostingClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        GaussianNB(),
        BernoulliNB(),
        KNeighborsClassifier(n_neighbors=5),
        MLPClassifier(max_iter=500, random_state=42),
        MultinomialNB(),
        RandomForestClassifier(random_state=42),
        LinearSVC(random_state=42),
        BaggingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        ExtraTreesClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),

        QuadraticDiscriminantAnalysis(reg_param=0.1),
        LinearDiscriminantAnalysis(),
        VotingClassifier(estimators=[('logistic', LogisticRegression(max_iter=500, random_state=42)),
                                     ('mlp', MLPClassifier(max_iter=500, random_state=42)),
                                     ('hist_gb', HistGradientBoostingClassifier(random_state=42)),
                                     ('gaussian_nb', GaussianNB())], voting='hard')
    ]
    results = []
    # Используем Parallel для распараллеливания обучения
    results = Parallel(n_jobs=-1)(
        delayed(train_classifier)(clf, X_train, y_train, X_val, y_val, X_test, y_test) for clf in classifiers)

    # Создаем DataFrame из результатов
    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df)
    # Сохраняем результаты в CSV файл
    results_df.to_csv('classification_results.csv', index=False)

    return results_df  # Возвращаем DataFrame с результатами

def eval_2(_df):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = split_scale_data(_df)
    # Список классификаторов для перебора
    classifiers = {
        'LGBMClassifier':LGBMClassifier(),  # Ok
        'CatBoostClassifier':CatBoostClassifier(silent=True),  # Ok
        'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis(),  # Ok
        'ExtraTreesClassifier':ExtraTreesClassifier(),  # Ok
        'LogisticRegression':LogisticRegression(max_iter=1000),  # Ok
        'DecisionTreeClassifier':DecisionTreeClassifier(),  # Ok
        'AdaBoostClassifier':AdaBoostClassifier(),  # Ok
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVC': SVC(kernel='linear'),  # Используем линейное ядро для SVC
        'XGBClassifier':XGBClassifier(use_label_encoder=False, eval_metric='logloss'), #No
        'GaussianNB':GaussianNB(), #No
        'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis(), #No
        'BaggingClassifier':BaggingClassifier(), #No
    }


    # Словарь для хранения выбранных признаков и их оценок
    selected_features = {}
    # Перебор классификаторов
    for name, model in classifiers.items():
        # Создание экземпляра RFECV
        selector = RFECV(model, step=1, cv=5)
        # Обучение модели
        selector.fit(X_train, y_train)
        # Получение выбранных признаков
        X_selected = selector.transform(X_train)
        # Сохранение результатов
        selected_features[name] = {
            'n_features': selector.n_features_,
            'support': selector.support_,
            'ranking': selector.ranking_,
            'X_selected': X_selected
        }
        print(f"{name}: {selected_features[name]['n_features']} выбранных признаков")
        results_df = pd.DataFrame(selected_features)
        results_df.to_csv('selected_features.csv', index=False)
        print(f"Results saved to {'selected_features.csv'}")


    # Теперь вы можете использовать X_selected для дальнейшего обучения или оценки моделей


def evaluate_classifiers_with_rfe(classifiers,_df, n_features_to_select=10):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = split_scale_data(_df)
    results = []  # Список для хранения результатов
    for clf in classifiers:
        print(f"Evaluating classifier: {clf.__class__.__name__}")
        rfe = RFE(estimator=clf, n_features_to_select=n_features_to_select) # Создание объекта RFE
        X_rfe = rfe.fit_transform(X_train, y_train) # Применение RFE
        selected_indices_rfe = rfe.get_support(indices=True)  # Получение индексов выбранных признаков
        selected_features = [feature_names[i] for i in selected_indices_rfe]
        print("Selected features by RFE:", selected_features)  # Вывод выбранных признаков

        clf.fit(X_rfe, y_train) # Обучение модели на отобранных признаках
        X_val_rfe = rfe.transform(X_val) # Для валидационной выборки также нужно отобрать признаки
        y_val_pred = clf.predict(X_val_rfe) # Оценка модели на валидационной выборке
        # Получение метрик для валидационной выборки
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        val_conf_matrix = confusion_matrix(y_val, y_val_pred)

        # Оценка модели на тестовой выборке
        X_test_rfe = rfe.transform(X_test)  # Для тестовой выборки также нужно отобрать признаки
        y_test_pred = clf.predict(X_test_rfe)  # Оценка модели на тестовой выборке

        # Получение метрик для тестовой выборки
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        # Сохранение результатов в список
        results.append({
            'Classifier': clf.__class__.__name__,
            'Selected Features': selected_features,
            'Validation Accuracy': val_report['accuracy'],
            'Validation Precision': val_report['weighted avg']['precision'],
            'Validation Recall': val_report['weighted avg']['recall'],
            'Validation F1 Score': val_report['weighted avg']['f1-score'],
            'Test Accuracy': test_report['accuracy'],
            'Test Precision': test_report['weighted avg']['precision'],
            'Test Recall': test_report['weighted avg']['recall'],
            'Test F1 Score': test_report['weighted avg']['f1-score'],
        })
        print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
        print("Validation Confusion Matrix:\n", val_conf_matrix)
        print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
        print("Test Confusion Matrix:\n", test_conf_matrix)
        print("\n" + "=" * 50 + "\n")
        results_df = pd.DataFrame(results)
        results_df.to_csv('results.csv', index=False)
        print(f"Results saved to {'results.csv'}")


def start_r(df):
    classifiers = [
        LGBMClassifier(),#Ok
        CatBoostClassifier(silent=True),#Ok
        LinearDiscriminantAnalysis(),#Ok
        ExtraTreesClassifier(),#Ok
        LogisticRegression(max_iter=1000),#Ok
        DecisionTreeClassifier(),#Ok
        AdaBoostClassifier(),#Ok
        RandomForestClassifier(),
        SVC(),
        #XGBClassifier(use_label_encoder=False, eval_metric='logloss'), #No
        #GaussianNB(), #No
        #QuadraticDiscriminantAnalysis(), #No
        #BaggingClassifier(), #No
    ]

    _df = df.copy()
    evaluate_classifiers_with_rfe(classifiers,df, n_features_to_select=15)
