import os
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score

from params import *
from plots import feat_importance_plot

def logreg(X, y, title, filename):
    # Permutation feature importance
    model = LogisticRegression(solver = 'newton-cg', random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)
    model.fit(X,y)
    results = permutation_importance(model, X, y, scoring = 'f1_weighted', random_state = 42)
    fig = feat_importance_plot(results, X)
    plt.title('\n' + title + '\n', fontsize = 16)
    plt.xlabel('Importance')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    plt.close(fig)

    # Model performance
    model.fit(X_train,y_train)
    print('Classification Report for logreg')
    print(classification_report(y_test, model.predict(X_test), target_names = ['F. palmarum', 'F. pennanti', 'F. sublineatus', 'F. tristriatus']))
    print('Confusion Matrix for logreg')
    print(confusion_matrix(y_test, model.predict(X_test)))

def dtc(X, y, title, filename):
    # Permutation feature importance
    model = DecisionTreeClassifier(random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)
    model.fit(X,y)
    results = permutation_importance(model, X, y, scoring = 'f1_weighted', random_state = 42)
    fig = feat_importance_plot(results, X)
    plt.title('\n' + title + '\n', fontsize = 16)
    plt.xlabel('Importance')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    plt.close(fig)

    # Model performance
    model.fit(X_train,y_train)
    print('Classification Report')
    print(classification_report(y_test, model.predict(X_test), target_names = ['F. palmarum', 'F. pennanti', 'F. sublineatus', 'F. tristriatus']))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, model.predict(X_test)))


def rtf(X, y, title, filename):
    # Permutation feature importance
    model = RandomForestClassifier(n_estimators=100, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)
    model.fit(X,y)
    results = permutation_importance(model, X, y, scoring = 'f1_weighted', random_state = 42)
    fig = feat_importance_plot(results, X)
    plt.title('\n' + title + '\n', fontsize = 16)
    plt.xlabel('Importance')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    plt.close(fig)

    # Model performance
    model.fit(X_train,y_train)
    print('Classification Report')
    print(classification_report(y_test, model.predict(X_test), target_names = ['F. palmarum', 'F. pennanti', 'F. sublineatus', 'F. tristriatus']))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, model.predict(X_test)))
