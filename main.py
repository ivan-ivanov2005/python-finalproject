import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn as sk
from pandas.core.groupby.groupby import DataError
import requests
import sqlite3
import json
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

sns.set()

st.title("Финальный проект по питону")

"""В данном проекте будет приведена информация о людях, их заработке и о том, как именно пол/образование/семейное положение и другие немаловажные факторы кореллируют
с уровнем дохода нашей тестирующей выборки. Также, постараемся использовать алгоритмы машинного обучения для того чтобы предсказать на обучающей выборке то, сколько человек будет зарабатывать в
зависимости от уже упомянтых нами факторов"""

df = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/adult_train.csv", sep=";")
df1 = df
"""Вот так выглядит наша выборка:"""
st.write(df.head())
show1 = st.sidebar.checkbox("Обработкой данных + pandas + numpy")
st.title("Часть с изначальной обработкой данных + продвинутое использование pandas")
if show1:
    """В данной части проанализируем возможно существующие корелляции между такими факторами, как доход и пол/доход и семейное положение,
    доход и возраст, доход и образование. Ну в общем, вы поняли...."""

    st.write("----------------------------------------------------------")
    """В данных графиках выведем общие данные, посмотрим на распределение полов, возрастов, женатых/неженатых и тд людей в нашей выборке. То есть, вывели общие данные по каждому критерию"""
    fig = plt.figure(figsize=(30, 20))
    cols = 5
    rows = int(df1.shape[1] / cols)
    for i, column in enumerate(df1.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df1.dtypes[column] == np.object:
            df1[column].value_counts().plot(kind="bar", axes=ax, color='green')
        else:
            df1[column].hist(axes=ax, color='green')
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    st.pyplot(plt, clear_figure=True)

    """Теперь построим интерактивный график, который будет показывать нам сколько людей в каком возрасте зарабатывают >50k и <50k"""
    """Но для начала переведем наш показатель пола и покатель дохода в бинарный вид, то есть если кто-то зарабатывает >50k, то у него Target=1, иначе 0. Это нужно для более 
    простого анализа в дальнейшем"""
    """Так же с полом, если женщина, то ей соответствует показатель 1, иначе 0"""
    # df.loc[df["Sex"] == " Male", "Sex"] = 0
    # df.loc[df["Sex"] == " Female", "Sex"] = 1
    data = df[df['Sex'] == " Female"][['Age', 'Sex', 'Target']].sort_values('Age')
    # st.write(data.head())
    pie_chart = df.groupby(['Age', 'Target']).size().reset_index(name='count')
    # st.write(pie_chart.head())
    age = st.slider('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()),
                     step=1)
    fig = px.pie(pie_chart.loc[(pie_chart['Age'] == age)], values='count',
                 names='Target', title=f'Distribution by age {age} without gender segregation')
    fig.update_traces(textposition='inside')
    st.plotly_chart(fig, use_container_width=True)

    st.write("----------------------------------------------------------")
    """Теперь проанализируем наши данные, если есть поправка на пол"""
    data_sex = df[['Age', 'Target', 'Sex']].sort_values('Age')
    # st.write(data_sex)
    age = st.slider('Choose the age:', 17, 90)
    fig, ax = plt.subplots()
    data_sex1 = data_sex[data_sex['Age'] == age]
    # st.write(data_sex1.head())
    chart = sns.countplot(y='Sex', hue="Target", data=data_sex1, ax=ax, palette='magma')
    chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
    chart.set_title(f"Distribution by {age}")
    chart.set(xlabel='Number of people', ylabel='Factors')
    st.pyplot(fig)

show2 = st.sidebar.checkbox("Обработка данных с помощью API + SQL + использование незнакомой библиотеки")
st.title("Обработка данных с помощью API + SQL + использование незнакомой библиотеки")
if show2:
    st.write("В данном разделе выведем корелляцию между уровнем бедности в стране и качестве жизни в ней")
    """Сперва попробуем высчитать уровень бедности, основываясь на нашей выборке. Понятно, что данные будут скорее не самыми показательными
    в силу неполной выборки, однако можем посчитать примерное распределение"""
    """Для того чтобы в нашей выборке было меньше выбросов, которые могли бы делать аналитику менее достоверное, уберем 
    тех, у которых перцентиль по возрасту строго меньше 2.5% или строго больше 97.5%, для того чтобы не делать поправки на слишком молодых/слишком старых"""
    df_filtered = df[(df["Age"] > df["Age"].quantile(0.025))
        & (df["Age"] < df["Age"].quantile(0.975))]
    """Посмотрим, сколько данных мы убрали"""
    num_of_removed = df_filtered.shape[0]  / df.shape[0]
    f"""У нас осталось {num_of_removed} от нашей изначальной выборки"""
    """Теперь рассмотрим, респонденты из каких страны рассматриваются в данном исследовании"""
    all_the_countries = df_filtered["Country"].unique()
    num_of_countries = len(all_the_countries)
    f"""{num_of_countries} - количество разных стран, откуда происходят респонденты. Приведем пример пары таких стран:"""
    for i in range(4):
        st.write(all_the_countries[i])
    """Продолжение в питоновском ноутбуке"""

show3 = st.sidebar.checkbox("Использование geopandas + sql")
st.title("Использование geopandas для более наглядной визуализации")
if show3:
    """Смотрите ноутбук с кодом по этой теме в гитхабе - он с названием ivan_geopandas.ipynb"""

show4 = st.sidebar.checkbox("Использование моделей машинного обучения для предсказания на тестовой выборке уровня заработка")
st.title("Использование алгоритмов машинного обучения")
if show4:
    with st.echo(code_location='below'):
        """В данном блоке нам предстоит воспользоваться алгоритмами машинного обучения для того чтобы основываясь на DecisionTreeClassifier"""
        """Основная цель будет заключаться в том, что в тестовой дате есть пропущенные значения, которые мы должны заполнить при помощи fit predict, а также 
        вывести степень достоверности нашей полученной выборки"""
        training = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/adult_train.csv", sep=';')
        testing = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/adult_test.csv", sep=';')
        st.write("Изначальная тренировочная выборка: ")
        st.write(training.head())
        st.write("Изначальная тестовая выборка: ")
        st.write(testing.head())
        testing = testing[
            (testing["Target"] == " >50K.") | (testing["Target"] == " <=50K.")
            ]

        # encode target variable as integer
        training.loc[training["Target"] == " <=50K", "Target"] = 0
        training.loc[training["Target"] == " >50K", "Target"] = 1

        testing.loc[testing["Target"] == " <=50K.", "Target"] = 0
        testing.loc[testing["Target"] == " >50K.", "Target"] = 1
        testing["Age"] = testing["Age"].astype(int)
        categories = [
            x for x in training.columns if training[x].dtype.name == "object"
        ]
        numericals = [
            x for x in training.columns if training[x].dtype.name != "object"
        ]
        for x in categories:
            training[x].fillna(training[x].mode()[0], inplace=True)
            testing[x].fillna(training[x].mode()[0], inplace=True)

        for c in numericals:
            training[c].fillna(training[c].median(), inplace=True)
            testing[c].fillna(training[c].median(), inplace=True)
        training = pd.concat(
            [training[numericals], pd.get_dummies(training[categories])],
            axis=1,
        )

        testing = pd.concat(
            [testing[numericals], pd.get_dummies(testing[categories])],
            axis=1,
        )
        testing["Country_ Holand-Netherlands"] = 0 #там баг в данных
        X_train = training.drop(["Target"], axis=1)
        y_train = training["Target"]

        X_test = testing.drop(["Target"], axis=1)
        y_test = testing["Target"]
        tree = DecisionTreeClassifier(max_depth=4, random_state=19)
        tree.fit(X_train, y_train)
        tree_predictions = tree.predict(X_test[X_train.columns])
        a = accuracy_score(y_test, tree_predictions)
        st.write("Степень достоверности наших данных, которыми мы заполнили выборку: ")
        st.write(a)
        # st.write(testing.head())

show5 = st.sidebar.checkbox("Web scrapping и Api methods")
st.title("Обработка данных с помощью web scrapping и api-methods")
if show5:
    """С помощью нашей карты смогли увидеть, что страны с самым высоким коэффициентом Джини - страны Африки. Докажем нашу гипотезу в питоновском ноутбуке на гитхабе, который называется - api.ipynb"""
