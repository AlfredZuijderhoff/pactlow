import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle
import json

# Chemin des fichiers
csv_file = 'recettes.csv'
model_file = 'model.pkl'
background_image = 'https://images.pexels.com/photos/1287145/pexels-photo-1287145.jpeg?cs=srgb&dl=pexels-eberhardgross-1287145.jpg&fm=jpg'

# Ajouter une image de fond
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialiser les états de session pour stocker les ingrédients ajoutés
if 'ingredients_list' not in st.session_state:
    st.session_state['ingredients_list'] = []
if 'prediction_ingredients_list' not in st.session_state:
    st.session_state['prediction_ingredients_list'] = []

# Personnalisation de l'interface
st.markdown(
    """
    <style>
    .stApp {{
        font-family: "Arial", sans-serif;
    }}
    .stSidebar {{
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
    }}
    .stButton>button {{
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }}
    .stButton>button:hover {{
        background-color: #0056b3;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Vérifier si le fichier CSV existe, sinon créer un DataFrame vide
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    data = pd.DataFrame(columns=['nom_recette', 'ingredients', 'temps_deshydratation', 'quantite_eau'])

# Fonction pour encoder les ingrédients
def encode_ingredients(df, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
        ingredients_encoded = mlb.fit_transform(df['ingredients'].apply(eval))
        return pd.DataFrame(ingredients_encoded, columns=mlb.classes_), mlb
    else:
        ingredients_encoded = mlb.transform(df['ingredients'].apply(eval))
        return pd.DataFrame(ingredients_encoded, columns=mlb.classes_)

# Entraînement du modèle si les données sont présentes
mlb = None
if not data.empty:
    ingredients_encoded, mlb = encode_ingredients(data)
    X = pd.concat([ingredients_encoded, data[['temps_deshydratation']]], axis=1)
    y = data['quantite_eau']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
else:
    model = None
    mse = None

# Interface Streamlit
st.title("Lo - Prédiction de la RéH des REC")

# Ajout de recettes
st.sidebar.header("Ajouter une nouvelle recette")

nom_recette = st.sidebar.text_input("Nom de la recette")
ingredient = st.sidebar.text_input("Ingrédient")
quantite = st.sidebar.number_input("Quantité (en grammes)", min_value=0, key='quantite')

if st.sidebar.button("Ajouter cet ingrédient"):
    if ingredient and quantite:
        st.session_state['ingredients_list'].append((ingredient, quantite))
        st.sidebar.write(f"Ingrédient ajouté : {ingredient} - {quantite}g")
    else:
        st.sidebar.error("Veuillez remplir tous les champs d'ingrédient et de quantité.")

# Afficher les ingrédients ajoutés
if st.session_state['ingredients_list']:
    st.sidebar.write("Ingrédients ajoutés :")
    for ing, qty in st.session_state['ingredients_list']:
        st.sidebar.write(f"- {ing} : {qty}g")

temps_deshydratation = st.sidebar.number_input("Temps de déshydratation (en heures)", min_value=0)
quantite_eau = st.sidebar.number_input("Quantité d'eau nécessaire (en grammes)", min_value=0)

if st.sidebar.button("Ajouter recette"):
    if nom_recette and st.session_state['ingredients_list'] and temps_deshydratation and quantite_eau:
        ingredients_dict = dict(st.session_state['ingredients_list'])
        new_data = pd.DataFrame({
            'nom_recette': [nom_recette],
            'ingredients': [json.dumps(ingredients_dict)],
            'temps_deshydratation': [temps_deshydratation],
            'quantite_eau': [quantite_eau]
        })

        # Ajouter la nouvelle recette aux données existantes
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(csv_file, index=False)

        # Réentraîner le modèle avec les nouvelles données
        ingredients_encoded, mlb = encode_ingredients(data, mlb=mlb)
        X = pd.concat([ingredients_encoded, data[['temps_deshydratation']]], axis=1)
        y = data['quantite_eau']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.session_state['ingredients_list'] = []  # Réinitialiser la liste des ingrédients après l'ajout
        st.success("Recette ajoutée et modèle réentraîné avec succès!")
    else:
        st.error("Veuillez remplir tous les champs")

# Prédiction
st.sidebar.header("Prédire la quantité d'eau")

pred_ingredient = st.sidebar.text_input("Ingrédient pour prédiction")
pred_quantite = st.sidebar.number_input("Quantité de l'ingrédient (en grammes)", min_value=0, key='pred_quantite')

if st.sidebar.button("Ajouter cet ingrédient à la prédiction"):
    if pred_ingredient and pred_quantite:
        st.session_state['prediction_ingredients_list'].append((pred_ingredient, pred_quantite))
        st.sidebar.write(f"Ingrédient ajouté : {pred_ingredient} - {pred_quantite}g")
    else:
        st.sidebar.error("Veuillez remplir tous les champs d'ingrédient et de quantité pour la prédiction.")

# Afficher les ingrédients ajoutés pour la prédiction
if st.session_state['prediction_ingredients_list']:
    st.sidebar.write("Ingrédients pour la prédiction :")
    for ing, qty in st.session_state['prediction_ingredients_list']:
        st.sidebar.write(f"- {ing} : {qty}g")

input_temps = st.sidebar.number_input("Temps de déshydratation pour prédiction (en heures)", min_value=0)

if st.sidebar.button("Prédire"):
    if model and st.session_state['prediction_ingredients_list'] and input_temps:
        input_ingredients_dict = dict(st.session_state['prediction_ingredients_list'])
        input_df = pd.DataFrame([json.dumps(input_ingredients_dict)], columns=['ingredients'])
        input_encoded = encode_ingredients(input_df, mlb=mlb)  # Correction: on attend une seule valeur
        input_data = pd.concat([input_encoded, pd.DataFrame({'temps_deshydratation': [input_temps]})], axis=1)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)[0]
        st.write(f"Quantité d'eau prédite: {prediction:.2f} grammes")
    else:
        st.warning("Modèle non entraîné ou données manquantes.")

# Afficher l'erreur quadratique moyenne s'il y en a une
if mse is not None:
    st.write(f"Erreur quadratique moyenne: {mse:.2f}")

# Visualisation des données sous forme de tableau
st.header("Tableau des recettes")

if not data.empty:
    st.write(data[['nom_recette', 'temps_deshydratation', 'quantite_eau']])
else:
    st.write("Aucune donnée disponible.")







