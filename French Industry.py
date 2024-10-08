#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:48:35 2024

@author: morganemuller
"""

import sys
sys.path.append('src')

import streamlit as st
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics import silhouette_samples, silhouette_score

# Import the data
df=pd.read_csv("df.csv")

st.title("French Industry Project")

###SIDEBAR ###
# Add the project logo 
logo_url = "logo.png"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Contexte et objectifs du projet", "Le Jeu de Données", "Analyses statistiques", "Modélisation", "Analyse des Résultats", "Conclusions & Perspectives"]
page=st.sidebar.radio("Sélectionnez une partie :", pages)

# Add a line before the section
st.sidebar.markdown("---") 

# Adding logo, names, and LinkedIn links with photos
st.sidebar.markdown("## Project Team")


# Add team members
team_members = [
    {"name": "Morgane MULLER", "photo_url": "Morgane.jpeg", "linkedin_url": "https://www.linkedin.com/in/morgane-muller/"},
    {"name": "Nolwenn RUMEUR", "photo_url": "Nolwenn.jpeg", "linkedin_url": "https://www.linkedin.com/in/nolwennrumeur/"},
    {"name": "Leila BENAMEUR", "photo_url": "Leila.jpeg", "linkedin_url": "https://www.linkedin.com/in/le%C3%AFla-benameur-45292325/"},
    {"name": "Samir BELFEROUM", "photo_url": "Samir.jpeg", "linkedin_url": "https://www.linkedin.com/in/sbelferoum/"},
]

for member in team_members:
    # Create a layout with image and text side by side
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.image(member['photo_url'])
    with col2:
        st.markdown(f"**{member['name']}**")
        st.markdown(f"[LinkedIn]({member['linkedin_url']})")

# Add a line after the section
st.sidebar.markdown("---")

## Introduction Page 
if page == pages[0]: 
    st.write("### Contexte et objectifs du projet")
    st.markdown("""
              #### Contexte  
              Notre projet d’analyse de données porte sur la comparaison des inégalités en France, un sujet vaste et sur lequel de nombreux organismes et chercheurs travaillent déjà actuellement. 

              Nous avons choisi de nous concentrer sur les inégalités de salaire sur le territoire français en posant la problématique suivante :
              """)
    st.write("**Dans quelle mesure la géographie influence-t-elle les inégalités de salaires en France ?**")
    st.write("**La géographie**, dans le cadre de cette problématique, concerne l'étude des facteurs spatiaux et régionaux qui influencent les conditions économiques et sociales des populations et des entreprises. Elle examine comment les caractéristiques physiques, les infrastructures, et les dynamiques locales, notamment économiques, impactent le développement et les niveaux de salaire.")
    st.write("**Les inégalités de salaires**, elles, se réfèrent aux différences de rémunération perçues par les travailleurs en fonction de divers facteurs tels que l'expérience, l'éducation, le secteur d'activité, le genre, et, dans ce cadre, la localisation géographique. Ces disparités peuvent se manifester à travers des écarts de salaire brut, des primes, et des avantages sociaux.")
              
    st.write("Ce projet est réalisé dans le cadre de la formation Data Analyst de DataScientest et ne s’insère pas directement dans les métiers respectifs des membres du groupe. Il s’agit pour nous d’une première, et nous sommes enthousiastes à l’idée de découvrir de près le métier de data analyst. En revanche, ce projet pourrait facilement s’insérer dans le métier d’un data analyst travaillant pour une institution publique (par exemple l’Observatoire des inégalités, une collectivité territoriale voulant en connaître plus sur les inégalités de salaire dans sa localité, ou encore un ministère - à titre d’exemple le ministère de la Cohésion des territoires ou le ministère du Travail)  souhaitant proposer des politiques publiques pour réduire ces inégalités.")

    st.write("**Du point de vue technique**, les données sont en effet collectées et traitées dans un premier temps par l’INSEE, avant d’être analysées statistiquement et modélisées / visualisées par des agents de cet institut ou d’un autre organisme public.") 
    st.write("**Du point de vue économique**, les organismes publics mentionnées ci-dessus ont besoin de réaliser des analyses des impacts économiques de ces inégalités afin de proposer de nouvelles politiques publiques visant à réduire ces dernières avant enfin d’évaluer le résultat de ces mêmes politiques publiques.")
    st.write("**Du point de vue scientifique**, il serait intéressant de réaliser une revue de littérature car de très nombreuses études et recherches existent déjà sur le sujet. Cela nous permettrait de nous appuyer sur des travaux existants pour affiner nos recherches et nos hypothèses. A l’issue de nos travaux, il serait envisageable de diffuser nos résultats à la communauté scientifique et au grand public, via un rapport ou un article.")
                  
    st.markdown("""
            #### Les objectifs de notre analyse
            """)

    st.write("Les principaux objectifs que nous nous sommes fixés sont les suivants :") 
    st.write("**- Identifier les disparités salariales horaires régionales : cartographier les inégalités de salaires horaires à travers les différentes régions de France en utilisant des données géographiques et salariales détaillées.**")
    st.write("**- Analyser les facteurs déterminants des inégalités : déterminer les principaux facteurs géographiques, économiques, sociaux et individuels qui influencent les variations des salaires horaires, tels que l'accès aux infrastructures, le genre, l’âge, etc.**")
    st.write("**- Faire des propositions de politiques publiques pouvant participer à la réduction de ces inégalités au vu des conclusions de notre étude.**")

    st.write("Les différents membres du groupe ne sont pas experts des problématiques liées aux inégalités de salaire en France, mais sont curieux d’en apprendre plus sur ce sujet qui concerne directement tous les citoyens. Des projets similaires sont déjà menés par différents organismes publics et les principaux facteurs des inégalités de salaires en France sont déjà connus. Nous ne prétendons pas découvrir de nouveaux facteurs d’inégalités, mais bien comprendre de l’intérieur comment des data analysts ont pu trouver les réponses à ces questions cruciales pour les politiques publiques en France, et pourquoi pas réfléchir à des idées de politiques publiques de réduction des inégalités qui pourraient par la suite s’insérer dans les métiers respectifs des membres de l’équipe.") 
          

## Le Jeu De Données Page 
if page == pages[1]:
    st.write("### Le jeu de données")
    st.write("Les sources employées pour ce projet proviennent de l’INSEE, Institut national de la statistique et des études économiques ainsi que Data.gouv, qui est une plateforme officielle de l'État français dédiée à la diffusion et au partage des données publiques. Il s'agit donc de données officielles concernant les différents aspects économiques, démographiques et géographiques du territoire français.")

    st.markdown("""
    ### Compréhension et manipulation des données
    """)

    st.write("Afin d’atteindre les objectifs de notre projet, nous avons utilisé plusieurs jeux de données. De façon générale, ces derniers sont issus de l’INSEE ou de site gouvernementaux. L’INSEE (Institut National de la Statistique et des Etudes Economiques) est l'institut officiel français qui collecte des données de tous types sur le territoire français. Les données que ce dernier collecte et analyse sont disponibles librement, afin de permettre à tous les organismes publics, citoyens et entreprises privées d’accéder à des informations et analyses intéressantes qui peuvent les concerner.")

    st.write("Les jeux de données utilisés sont les suivants :")
    
    st.write("- La table **'base_etablissement_par_tranche_effectif.csv'** : Informations sur le nombre d'entreprises dans chaque ville française classées par taille.")
    st.write("- La table **'name_geographic_information'**: Données géographiques sur les villes françaises (principalement la latitude et la longitude, mais aussi les codes et les noms des régions/départements).")
    st.write("- La table **'net_salary_per_town_categories'** : Salaires par villes française par catégories d'emploi, âge et sexe. Ce fichier fournit des données sur les salaires nets horaires. Les indicateurs sont ventilés selon le sexe, l'âge, la catégorie socioprofessionnelle (hors agriculture).")
    st.write("- La table **'population.csv'** : Informations démographiques par ville, âge, sexe et mode de vie.")
    st.write("- La table **'referentiel-gares-voyageurs.csv'** : Informations sur les gares TGV présentes sur le territoire français.")
    st.write("- La table **'aeroports_france'** : Informations sur les aéroports français.")
    
    st.markdown("### Pertinence")

    st.write("Les variables qui nous semblent les plus pertinentes au regard de nos objectifs sont celles concernant les salaires, les tailles d’entreprises et les données géographiques. Le concept de variable cible n'est pas directement applicable au clustering, car il n'y a pas de variable spécifique à prédire. Lorsque nous effectuerons le clustering, nous pourrons analyser les caractéristiques des clusters obtenus et identifier les variables qui sont les plus pertinentes pour distinguer les différents groupes. Nous cherchons à découvrir des structures ou des modèles cachés dans les données en regroupant les observations similaires.")

    st.write("Notre jeu de données possède la particularité de comprendre un très grand nombre de lignes (notamment le fichier population.csv), ce qui peut compliquer la fusion avec les autres fichiers. Nous serons donc potentiellement amenés à travailler sur des portions de ce fichier dans le cadre d’études ciblées plutôt qu’avec les données dans leur ensemble. Nous avons été limités par les données géographiques car celles-ci ne contenaient pas toutes les latitudes et longitudes nécessaires à nos analyses. Cependant, nous avons pu les compléter avec d’autres données issues d’un site gouvernemental, ce qui nous a permis de surmonter ce problème.")


    st.write("#### Afficher les données")
    st.dataframe(df.head())
    
    st.write("#### Description des principales colonnes")

    st.write("**CODGEO** : Code unique pour chaque commune.")
    st.write("**total_entreprise**: Nombre total d'entreprises dans la commune.")
    st.write("**code_région** : Code de la région à laquelle appartient la commune.")
    st.write("**nom_région**: Nom de la région (par exemple, Rhône-Alpes.")
    st.write("**numéro_département** : Numéro du département auquel appartient la commune.")
    st.write("**nom_département** : Nom du département (par exemple, Ain).")
    st.write("**nom_commune** : Nom de la commune (par exemple, Ambérieu-en-Bugey).")
    st.write("**latitude et longitude**: Coordonnées géographiques de la commune.")
    st.write("**Prixm2Moyen** : Prix moyen du mètre carré dans la commune.")
    st.write("**aeroport**: Indicateur (0 ou 1) indiquant la présence d'un aéroport à proximité.")
    st.write("**Gare**: Indicateur (0 ou 1) indiquant la présence d'une gare.")
    st.write("**No_diplome** : Nombre de personnes sans diplôme dans la commune.")
    st.write("**BEPC_brevet_DNB** : Nombre de personnes ayant obtenu le brevet (ou équivalent).")
    st.write("**CAP_BEP**: Nombre de personnes ayant obtenu un CAP ou un BEP.")
    st.write("**Bac_Brevet_pro** : Nombre de personnes ayant obtenu un bac professionnel ou un brevet professionnel.")
    st.write("**Bac2** : Nombre de personnes ayant un diplôme de niveau Bac+2.")
    st.write("**Bac3_Bac4** : Nombre de personnes ayant un diplôme de niveau Bac+3 ou Bac+4.")
    st.write("**Bac5**: Nombre de personnes ayant un diplôme de niveau Bac+5 ou plus.")
    
## Analyses statistiques Page ##
if page == pages[2] : 
    st.write("### Analyses statistiques")
    
    choix = [
        'Test ANOVA sur les prix au m² entre les différentes régions', 
        'Régression linéaire', 
        ]
    
    # Interface utilisateur
    choix = ['TEST ANOVA', 'Régression linéaire', 'Hierarchical clustering avant PCA', 'Hierarchical clustering après PCA']  # Ajoutez d'autres options si nécessaire
    option = st.selectbox("Choix", choix)
    st.write('Le test statistique choisi est :', option)

    if option == 'TEST ANOVA':
    
       # Filtrer les colonnes nécessaires pour l'analyse
       df_region_price = df[['nom_région', 'Prixm2Moyen']].dropna()
    
       # Obtenir les groupes de prix par région
       regions = df_region_price['nom_région'].unique()
       grouped_data = [df_region_price[df_region_price['nom_région'] == region]['Prixm2Moyen'] for region in regions]
    
       # Effectuer une ANOVA (analyse de la variance) pour comparer les moyennes entre les régions
       anova_result = stats.f_oneway(*grouped_data)
    
       # Afficher le résultat du test ANOVA avec Streamlit
       st.write(f"Statistique F : {anova_result.statistic}")
       st.write(f"P-value : {anova_result.pvalue}")

       st.write("Les différences de prix au m² entre les régions sont statistiquement significatives.")
       
       st.write("**Interprétation :**")
       st.write("Statistique F élevée (190.91) : Cela signifie qu'il y a une grande différence entre les moyennes des prix au m² dans les différentes régions par rapport à la variance à l'intérieur de chaque région. Plus la statistique F est grande, plus la différence entre les groupes est importante.")
       st.write("P-value de 0.0 : Une p-value de 0.0 signifie que la probabilité d'observer ces différences par hasard est extrêmement faible. Cela nous permet de rejeter l'hypothèse nulle, qui stipule qu'il n'y a pas de différence entre les prix au m² des différentes régions.")

       st.write("**Conclusion :**")
       st.write("Les différences entre les prix moyens au m² des différentes régions sont statistiquement significatives. Autrement dit, le prix de l'immobilier varie de manière significative d'une région à l'autre. Nous pouvons en conclure que les régions n'ont pas des prix immobiliers homogènes et que certains facteurs régionaux influencent les prix au m².")
        
    if option == 'Régression linéaire':
       # Sélection des colonnes pertinentes pour la régression
       regression_columns = [
       'total_entreprise', 'Prixm2Moyen', 'total_population'
        ]

       # Filtrer les données et supprimer les valeurs manquantes
       df_regression = df[regression_columns].dropna()

       # Variables explicatives (indépendantes) et variable cible (dépendante)
       X = df_regression.drop(columns=['total_entreprise'])  # Variables explicatives
       y = df_regression['total_entreprise']  # Variable cible

       # Normalisation des variables explicatives
       scaler = StandardScaler()
       X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  
       # Ajouter une constante pour l'intercept
       X_scaled = sm.add_constant(X_scaled)

       # Calcul du VIF (Variance Inflation Factor) pour chaque variable
       vif_data = pd.DataFrame()
       vif_data["Variable"] = X_scaled.columns
       vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]

       st.write("Variance Inflation Factors (VIF) :")
       st.write(vif_data)

        # Supprimer les variables avec un VIF élevé (> 10)
       variables_to_drop = vif_data[vif_data["VIF"] > 10]["Variable"]
       X_scaled = X_scaled.drop(columns=variables_to_drop)

       st.write(f"Variables supprimées pour colinéarité : {variables_to_drop.tolist()}")

        # Ajuster le modèle de régression linéaire
       model = sm.OLS(y, X_scaled).fit()

       # Ajout des nouvelles colonnes pour la régression
       new_columns = ['aeroport', 'Gare']

       # Mettre à jour les données pour inclure les nouvelles variables
       df_regression = df_regression.join(df[new_columns], how='inner').dropna()

       # Variables explicatives mises à jour
       X_new = df_regression.drop(columns=['total_entreprise'])

       # Normalisation des nouvelles données
       X_new_scaled = pd.DataFrame(scaler.fit_transform(X_new), columns=X_new.columns)

       # Ajouter la constante
       X_new_scaled = sm.add_constant(X_new_scaled)

       # Ajustement du nouveau modèle
       model_new = sm.OLS(y, X_new_scaled).fit()

       # Résumé du modèle
       print(model_new.summary())
       
       st.write("L'analyse des résultats de la régression OLS montre R² très élevé (0.999), ce qui indique que le modèle explique presque toute la variabilité de la variable dépendante, total_entreprise. Les qualifications éducatives (comme BEPC_brevet_DNB, Bac_Brevet_pro, et Bac5) ont un impact positif significatif, tandis que l'absence de diplôme (No_diplome) a un effet négatif. Cependant, la présence d'une forte multicolinéarité soulève des préoccupations sur la robustesse des coefficients individuels, rendant difficile l'évaluation précise de chaque variable. Certaines variables, telles que total_population et Moyenne_entreprise, montrent également des coefficients négatifs, indiquant qu'une augmentation de ces facteurs peut réduire la valeur totale des entreprises.")
       st.write("En résumé, bien que le modèle ait une bonne capacité explicative, des précautions doivent être prises concernant la multicolinéarité et l'interprétation des coefficients.")

## Modélisation Page ##
if page == pages[3] : 
   st.write("### Modélisation")
    
   choix = [
        'Clustering avant ACP', 
        'Clustering après ACP', 
    ]
    
    # Normalisation des données
    # Liste des colonnes à normaliser
   columns_to_scale = columns_to_scale = ['total_entreprise','Prixm2Moyen','total_population','Micro_entreprise',
    'Petite_entreprise','Moyenne_entreprise','Grande_entreprise',
    'Salaire_net_moyen','Salaire_net_moyen_heure_cadre','Salaire_net_moyen_heure_cadre_moyen','Salaire_net_moyen_heure_employé',
    'Salaire_net_moyen_heure_ouvrier','Salaire_net_moyen_heure_femme','Salaire_net_moyen_heure_cadre_femme','Salaire_net_moyen_heure_cadre_moyen_femme',
    'Salaire_net_moyen_heure_employé_femme','Salaire_net_moyen_heure_ouvrier_femme','Salaire_net_moyen_homme',
    'Salaire_net_moyen_heure_cadre_homme','Salaire_net_moyen_heure_cadre_moyen_homme',
    'Salaire_net_moyen_heure_employé_homme','Salaire_net_moyen_heure_ouvrier_homme','Salaire_net_moyen_heure_18_25',
    'Salaire_net_moyen_heure_26_50','Salaire_net_moyen_heure_plus_50','Salaire_net_moyen_heure_18_25_femme','Salaire_net_moyen_heure_26_50_femme',
    'Salaire_net_moyen_heure_plus_50_femme','Salaire_net_moyen_heure_18_25_homme','Salaire_net_moyen_heure_26_50_homme',
    'Salaire_net_moyen_heure_plus_50_homme',
    'No_diplome','BEPC_brevet_DNB','CAP_BEP','Bac_Brevet_pro','Bac2','Bac3_Bac4','Bac5']

    # Normaliser les colonnes sélectionnées
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df[columns_to_scale])

    # Remettre les colonnes normalisées dans un DataFrame avec les noms originaux
   df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)

    # Afficher les premières lignes du dataframe
   st.write("Affichage des premières lignes du dataframe :")
   st.dataframe(df_scaled.head())
    
    # Méthode du coude 
   st.write("Méthode du coude")
    # Afficher l'image 
   image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Méthode du coude avant ACP.png"  
   st.image(image_path, caption='Méthode du coude avant ACP', use_column_width=True)
   st.write("D'après la méthode du coude, le nombre optimal de clusters semble être 3.")

    # Méthode du dendrogramme
   st.write("Méthode du dendrogramme")
    # Afficher l'image 
   image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/ Méthode du dendrogramme avant ACP.png"  
   st.image(image_path, caption='Méthode du dendrogramme avant ACP', use_column_width=True)
   st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être 3ou 4.")

    # Algoritme des KMeans 
    # Appliquer le KMeans sur les colonnes normalisées
   kmeans = KMeans(n_clusters=3, random_state=42)
   kmeans.fit(df_scaled)

    #Afficher les centroîdes 
   st.write("Les centres des clusters sont :")
   st.write(kmeans.cluster_centers_)
    # Créer un scatter plot des points de données et des centres de clusters
   fig, ax = plt.subplots()
   ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', label="Données")
   ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centres des clusters")
    # Afficher le graphique dans Streamlit
   st.pyplot(fig)
    
    #Evaluation de la qualité du clustering
    # Calculer le Score de Silhouette
   silhouette_avg = silhouette_score(df_kmeans, kmeans.labels_)
    # Afficher le Score de Silhouette dans Streamlit
   st.write("Silhouette Score :", silhouette_avg)
    
   st.write("Taille du cluster 1 : 800")
   st.write("Taille du cluster 2 : 3861")
   st.write("Taille du cluster 3 : 1")
   st.write("### Modélisation")

   choix = [
         'Clustering sans ACP', 
         'Clustering après ACP', 
            ]

       # Interface utilisateur
   choix = ['Clustering sans ACP', 'Clustering après ACP']
   option = st.selectbox("Choix", choix)
   st.write('Le test statistique choisi est :', option)

   if option == 'Clustering sans ACP':
    
          # Liste des colonnes à normaliser
          columns_to_scale = columns_to_scale = ['total_entreprise','Prixm2Moyen','total_population','Micro_entreprise',
         'Petite_entreprise','Moyenne_entreprise','Grande_entreprise',
         'Salaire_net_moyen','Salaire_net_moyen_heure_cadre','Salaire_net_moyen_heure_cadre_moyen','Salaire_net_moyen_heure_employé',
         'Salaire_net_moyen_heure_ouvrier','Salaire_net_moyen_heure_femme','Salaire_net_moyen_heure_cadre_femme','Salaire_net_moyen_heure_cadre_moyen_femme',
         'Salaire_net_moyen_heure_employé_femme','Salaire_net_moyen_heure_ouvrier_femme','Salaire_net_moyen_homme',
         'Salaire_net_moyen_heure_cadre_homme','Salaire_net_moyen_heure_cadre_moyen_homme',
         'Salaire_net_moyen_heure_employé_homme','Salaire_net_moyen_heure_ouvrier_homme','Salaire_net_moyen_heure_18_25',
         'Salaire_net_moyen_heure_26_50','Salaire_net_moyen_heure_plus_50','Salaire_net_moyen_heure_18_25_femme','Salaire_net_moyen_heure_26_50_femme',
         'Salaire_net_moyen_heure_plus_50_femme','Salaire_net_moyen_heure_18_25_homme','Salaire_net_moyen_heure_26_50_homme',
         'Salaire_net_moyen_heure_plus_50_homme',
         'No_diplome','BEPC_brevet_DNB','CAP_BEP','Bac_Brevet_pro','Bac2','Bac3_Bac4','Bac5']

          # Normaliser les colonnes sélectionnées
          scaler = StandardScaler()
          df_scaled = scaler.fit_transform(df[columns_to_scale])

          # Remettre les colonnes normalisées dans un DataFrame avec les noms originaux
          df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)
          
          # Afficher les premières lignes de df_scaled
          st.write("Voici les premières lignes du DataFrame:")
          st.write(df_scaled.head())

          # Méthode du coude
          # Utiliser une image locale
image_path = "chemin/vers/ton/image.jpg"

# Afficher l'image
st.image(image_path, caption='Description de l\'image', use_column_width=True)


          # Méthode du dendogramme 
          # Effectuer un regroupement hiérarchique
          Z = linkage(df_scaled, method='ward', metric='euclidean')

          # Création de la figure avec Matplotlib
          fig, ax = plt.subplots(figsize=(20, 10))
          ax.set_title('Dendrogramme')

          # Tracer le dendrogramme avec des couleurs
          dendrogram(
          Z, 
          labels=df_scaled.index, 
          leaf_rotation=90., 
          color_threshold=290,  # Seuil pour changer les couleurs
          above_threshold_color='grey',  # Couleur des segments au-dessus du seuil
          ax=ax
          )

          # Affichage du graphique dans Streamlit
          st.pyplot(fig)
    
          # Algorithme des KMeans 
          # Appliquer le KMeans sur les colonnes normalisées
          kmeans = KMeans(n_clusters=3, random_state=42)
          kmeans.fit(df_scaled)

          # Affichage des coordonnées des centroïdes des clusters
          st.write("Coordonnées des centroïdes des clusters :")
          st.write(kmeans.cluster_centers_)

          # Création de la figure pour afficher les centroïdes
          fig, ax = plt.subplots()
          ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200)  # Les centroïdes

          # Affichage du graphique 
          st.pyplot(fig)

          # Créer un DataFrame pour les colonnes normalisées avec les labels des clusters
          df_kmeans = pd.DataFrame(df_scaled, columns=columns_to_scale)
          y_kmeans = kmeans.labels_

          # Création de la figure pour visualiser les clusters
          fig, ax = plt.subplots(figsize=(12, 12))

          # Visualisation des clusters
          ax.scatter(df_kmeans[y_kmeans == 0].iloc[:, 0], df_kmeans[y_kmeans == 0].iloc[:, 1], s=50, c='pink', label='Cluster 1')
          ax.scatter(df_kmeans[y_kmeans == 1].iloc[:, 0], df_kmeans[y_kmeans == 1].iloc[:, 1], s=50, c='blue', label='Cluster 2')
          ax.scatter(df_kmeans[y_kmeans == 2].iloc[:, 0], df_kmeans[y_kmeans == 2].iloc[:, 1], s=50, c='purple', label='Cluster 3')

          # Visualisation des centroïdes
          ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

          # Personnalisation des labels et du titre
          ax.set_title("Visualisation des clusters avec leurs centroïdes")
          ax.set_xlabel("Feature 1")
          ax.set_ylabel("Feature 2")
          ax.legend()

          # Affichage du graphique dans Streamlit
          st.pyplot(fig)
    
          # Evaluation de la qualité du clustering
          # Calculer le Silhouette Score
          silhouette_avg = silhouette_score(df_kmeans, kmeans.labels_)

          # Afficher le Silhouette Score dans Streamlit
          st.write("Silhouette Score :", silhouette_avg)
    
          # Définition des paramètres
          n_clusters = 3
          y_lower = 10

          # Effectuer le clustering avec KMeans
          kmeans = KMeans(n_clusters=n_clusters)
          kmeans.fit(df_scaled)

          # Calcul des coefficients de silhouette pour chaque point
          silhouette_vals = silhouette_samples(df_scaled, kmeans.labels_)

          # Calcul du coefficient de silhouette moyen
          silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)

          # Création d'un Silhouette Plot
          fig, ax = plt.subplots(figsize=(8, 6))
          ax.set_xlim([-0.1, 1])
          ax.set_ylim([0, len(df_scaled) + (n_clusters + 1) * 10])

          for i in range(n_clusters):
              # Extraction des valeurs des coefficients de silhouette pour le cluster i
             cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == i]
             cluster_silhouette_vals.sort()
    
           # Calcul de la taille du cluster
          size_cluster_i = cluster_silhouette_vals.shape[0]
          st.write(f"Taille du cluster {i+1} : {size_cluster_i}")
    
          # Calcul des limites supérieures pour le plot du cluster
          y_upper = y_lower + size_cluster_i
    
          # Tracé de la silhouette plot pour le cluster i
          color = plt.cm.get_cmap("inferno")(float(i) / n_clusters)
          ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)

          ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
          y_lower = y_upper + 10

          # Titre et légendes
          ax.set_title("Silhouette Plot des différents clusters")
          ax.set_xlabel("Valeurs des coefficients de silhouette")
          ax.set_ylabel("Cluster Label")

          # Tracé d'une ligne verticale correspondant au Silhouette Score moyen
          ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    
    # Afficher l'image 
   image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Silhouette score avant ACP.png  
   st.image(image_path, caption='Silhouette Plot des différents clusters', use_column_width=True)
   st.write("Nous obtenons un Silhouette score moyen (environ 0,5). Le Silhouette Plot nous montre que les clusters 1 et 2 sont plutôt bien définis, mais que la présence d'une ville unique dans le cluster 3 (Paris) tire vers le bas la valeur du Silhouette Score.")