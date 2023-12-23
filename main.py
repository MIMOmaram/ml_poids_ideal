import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# chagement de donnee
script_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(script_dir, 'ml1.csv'))

# n9asamhom 3ala features x et y
X = data[['sexe', 'taille']]
y = data['poids']

# X et y n9asamhom 3ala donnee d'entrainement w donnee de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Initialisation de modele de regresion lineaire
model = LinearRegression()
# entrainner le modele
model.fit(X_train, y_train)

# Fonction de prediction de poids
def predict_poids(sexe, taille):
    try:
        nouveau_poids = model.predict([[sexe, taille]])
        return f'Votre Poids Idéale: {nouveau_poids[0]:.2f} kg'
    except ValueError:
        return 'Veuillez entrer des valeurs valides pour le sexe et la taille.'

# interface utilisateur (b streamlit)
st.write('# Calculer Votre Poids Idéale')



# barre d'info supp
st.sidebar.subheader('Informations')
st.sidebar.write('Cette application utilise un modèle de régression linéaire pour prédire le poids en fonction du sexe et de la taille.')

# input user
st.header('Entrez les informations suivantes :')

# barre slider l taille
# Ajouter une image
taille = st.slider('Taille en cm:', min_value=100, max_value=220, value=170, step=1)

# selesction de sexe 0 F , 1 H
sexe = st.selectbox('Sexe', [0, 1], format_func=lambda x: 'Femme' if x == 0 else 'Homme')
# boutton de prediction
if st.button('Prédire Poids'):
    result_box = st.empty()
    result = predict_poids(sexe, taille)
    # Appliquer des styles de texte avec Markdown
    result_box.markdown(f'<p style="color: blue; font-size: 35px;">{result}</p>', unsafe_allow_html=True)
