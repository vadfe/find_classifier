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
from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def split_scale_data(df):
    _res = df.copy()
    drop_columns = ['p_close', 'p_30m_close', 'p_1h_close', 'p_open', 'p_30m_open', 'p_1h_open',
                    'p_high', 'p_30m_high', 'p_1h_high', 'p_low', 'p_30m_low', 'p_1h_low',
                    'close', 'open', 'high', 'low', 'volume', 'ma_prs', 'ma_vol', 'target']
    #Удаление    коррелирующих    признаков
    threshold = 0.9  # Порог корреляции
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
        LogisticRegression(max_iter=1000),
        SVC(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        LGBMClassifier(),
        CatBoostClassifier(silent=True),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        BaggingClassifier(),
        ExtraTreesClassifier()
    ]

    # Предполагается, что X_train, y_train, X_val, y_val уже определены
    evaluate_classifiers_with_rfe(classifiers,df, n_features_to_select=10)
