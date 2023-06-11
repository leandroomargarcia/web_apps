
'''Instrucciones como utilizar este codigo en la PC (no nse puede ejecutar desde COLAB)

1 - Debemos tener instalado en la PC python y Anaconda o visual studio code.
2 - Ejecutar desde la consola: pip install streamlit
3 - Navegar desde la consola hasta la carpeta en donde se encuentra este codigo en la PC.Ejecutar este codigo
4 - Si tira error de no se encuentra la libreria Streamlit.cli debemos reinstalar streamlit usando:
    1- pip unistall streamlit
    2- pip install streamlit
5 - Una vez ejecutado se abrirá una pagina web desde chrome con nuestra APP que se encuentra local.
'''

import subprocess

# Verificar si la biblioteca está instalada
try:
    import matplotlib.pyplot as plt
except ImportError:
    # La biblioteca no está instalada, se procede a instalarla
    subprocess.check_call(['pip', 'install', 'matplotlib'])

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve, RocCurveDisplay,precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import auc



st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms editable or poisonous?")
    st.sidebar.markdown("Are your mushrooms editable or poisonous?")

    @st.cache_resource()
    def load_data():
        data = pd.read_csv('C:/Users/O003132/Downloads/mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_resource()
    def split(df):
        y = df.type
        x = df.drop(['type'], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            y_pred = model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot()
            plt.title("Matriz de confusión")
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            y_scores = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            roc_disp.plot()
            plt.title("Curva ROC")
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            y_scores = model.predict_proba(x_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            pr_disp.plot()
            plt.title("Curva de Precisión y Recall")
            st.pyplot()
    
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['editable','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", 
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularización parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf","linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key= 'gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularización parameter)", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter,)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularización parameter)", 0.01, 10.0, step=0.01, key='C')
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = bool(st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap'))
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs= -1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)
    

if __name__ == '__main__':
    main()