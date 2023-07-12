import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Random Forest Classifier")

n_estimators = int(st.sidebar.number_input('Enter number of estimators', 0, 2000, 100, step = 10))

max_features = st.sidebar.selectbox(
    'Max Features', 
    ('auto', 'sqrt', 'log2', 'manual')
    )
if max_features == 'manual':
    max_features = int(st.sidebar.number_input('Max Features'))

max_samples = st.sidebar.slider('Max Samples', 1, X_train.shape[0], 150,step=25)

bootstrap_samples = st.sidebar.radio(
    "Bootstrap Samples",
    (True, False)
)
if bootstrap_samples == False:
    max_samples = None
# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred_tree = clf.predict(X_test)

    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap_samples,max_features=max_features, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    orig.empty()

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)
    labels1 = rf_clf.predict(input_array)


    col1, col2 = st.columns(2)
    with col1:
        st.header('Decision Tree Classifier')
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        orig = st.pyplot(fig)
        st.subheader("Accuracy of Decision Tree  " + str(round(accuracy_score(y_test, y_pred_tree),2)))
    with col2:
        st.header("RandomForest Classifier")
        ax1.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        ax1.contourf(XX, YY, labels1.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        orig1 = st.pyplot(fig1)
        st.subheader("Accuracy of Random Forest  " + str(round(accuracy_score(y_test, y_pred), 2)))