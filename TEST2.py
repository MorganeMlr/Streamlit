#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:02:50 2024

@author: morganemuller
"""

# # Diagnostic information
# st.write(f"Python executable: {sys.executable}")
# st.write(f"Python version: {sys.version}")
# st.write("Installed packages:")
# st.code(subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8"))

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
st.write("**Cartographier les inégalités : exploration des disparités salariales sur le territoire français par clustering géographique**")


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
 
             "")
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

       # Interface utilisateur
   choix = ['Clustering sans ACP', 'Clustering après ACP']
   option = st.selectbox("Choix", choix)
   st.write('Le test statistique choisi est :', option)

   if option == 'Clustering sans ACP':
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
      'No_diplome','BEPC_brevet_DNB','CAP_BEP','Bac_Brevet_pro','Bac2','Bac3_Bac4','Bac5'
      ]
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
      # Afficher les centroîdes 
      st.write("Les centres des clusters sont :")
      st.write(kmeans.cluster_centers_)
      # Créer un DataFrame pour les colonnes normalisées avec les labels des clusters
      df_kmeans = pd.DataFrame(df_scaled, columns=columns_to_scale)
      y_kmeans = kmeans.labels_
      # Visualisation des clusters
      plt.figure(figsize=(12, 12))
      plt.scatter(df_kmeans[y_kmeans == 0].iloc[:, 0], df_kmeans[y_kmeans == 0].iloc[:, 1], s=50, c='pink', label='Cluster 1')
      plt.scatter(df_kmeans[y_kmeans == 1].iloc[:, 0], df_kmeans[y_kmeans == 1].iloc[:, 1], s=50, c='blue', label='Cluster 2')
      plt.scatter(df_kmeans[y_kmeans == 2].iloc[:, 0], df_kmeans[y_kmeans == 2].iloc[:, 1], s=50, c='purple', label='Cluster 3')
      # Visualisation des centroïdes
      plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
      # Ajouter la légende
      plt.legend()
      st.pyplot(plt)
    
      # Evaluation de la qualité du clustering
      # Calculer le Score de Silhouette
      silhouette_avg = silhouette_score(df_kmeans, kmeans.labels_)
      # Afficher le Score de Silhouette dans Streamlit
      st.write("Silhouette Score :", silhouette_avg)
      st.write("Taille du cluster 1 : 800")
      st.write("Taille du cluster 2 : 3861")
      st.write("Taille du cluster 3 : 1")
      # Afficher l'image 
      image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Silhouette score avant ACP.png" 
      st.image(image_path, caption='Silhouette Plot des différents clusters', use_column_width=True)
      st.write("Nous obtenons un Silhouette score moyen (environ 0,5). Le Silhouette Plot nous montre que les clusters 1 et 2 sont plutôt bien définis, mais que la présence d'une ville unique dans le cluster 3 (Paris) tire vers le bas la valeur du Silhouette Score.")
   
      # Représentation graphique des villes présentent dans les différents clusters 
      image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Nuages de mots clusters.png" 
      st.image(image_path, caption='Nuages de mots des différents clusters', use_column_width=True)
   
   if option == 'Clustering après ACP':
      # Analyse des composantes 
      st.markdown("# Analyse des composantes")
      # Sélection des colonnes pour l'ACP
      columns_for_pca = [
       'total_population', 'Micro_entreprise', 'Petite_entreprise', 'Moyenne_entreprise', 'Grande_entreprise',
       'Salaire_net_moyen', 'Salaire_net_moyen_heure_cadre', 'Salaire_net_moyen_heure_cadre_moyen',
       'Salaire_net_moyen_heure_employé', 'Salaire_net_moyen_heure_ouvrier', 'Salaire_net_moyen_heure_femme',
       'Salaire_net_moyen_heure_cadre_femme', 'Salaire_net_moyen_heure_cadre_moyen_femme', 'Salaire_net_moyen_heure_employé_femme',
       'Salaire_net_moyen_heure_ouvrier_femme', 'Salaire_net_moyen_homme', 'Salaire_net_moyen_heure_cadre_homme',
       'Salaire_net_moyen_heure_cadre_moyen_homme', 'Salaire_net_moyen_heure_employé_homme', 'Salaire_net_moyen_heure_ouvrier_homme',
       'Salaire_net_moyen_heure_18_25', 'Salaire_net_moyen_heure_26_50', 'Salaire_net_moyen_heure_plus_50',
       'Salaire_net_moyen_heure_18_25_femme', 'Salaire_net_moyen_heure_26_50_femme', 'Salaire_net_moyen_heure_plus_50_femme',
       'Salaire_net_moyen_heure_18_25_homme', 'Salaire_net_moyen_heure_26_50_homme', 'Salaire_net_moyen_heure_plus_50_homme',
       'Prixm2Moyen', 'aeroport', 'Gare', 'No_diplome', 'BEPC_brevet_DNB', 'CAP_BEP', 'Bac_Brevet_pro', 'Bac2',
       'Bac3_Bac4', 'Bac5'
        ]
       # Supression des valeurs manquantes et vérification du type des données
      df_pca = df[columns_for_pca].dropna()
       # Standardisation des données
      scaler = StandardScaler()
      df_scaled = scaler.fit_transform(df_pca)
       # Utilisation de l'ACP pour expliquer 90% de la variance
      pca = PCA(n_components=0.9)
      df_pca_transformed = pca.fit_transform(df_scaled)
       # Explication du ratio de variance
      explained_variance_ratio = pca.explained_variance_ratio_
       # Affichage des résultats dans Streamlit
      st.write("Analyse en Composantes Principales (ACP)")
      st.write("Ratio de variance expliquée par chaque composante :")
      st.write(explained_variance_ratio)
      # Afficher l'image 
      image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Variance expliquée par composantes principales après ACP.png" 
      st.image(image_path, caption='Variance expliquée par composantes principales', use_column_width=True)
       
      # Méthode du coude 
      image_path = "//Users/morganemuller/Desktop/DATA/Streamlit/Méthode du coude après ACP.png" 
      st.image(image_path, caption='Méthode du coude après ACP', use_column_width=True)
      st.write("D'après la méthode du coude, le nombre optimal de clusters semble être 3.")
      
      # Méthode du dendrogramme
      image_path = "//Users/morganemuller/Desktop/DATA/Streamlit/Méthode du dendrogramme après ACP.png" 
      st.image(image_path, caption='Méthode du dendrogramme après ACP', use_column_width=True)
      st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être 3.")
      
      # Evaluation de la qualité du clustering
      # Défintiion du nombre de clusters basée sur les analyses précédentes
      n_clusters = 3
      # Application de l'algorithme des KMeans avec le nombre optimal de clusters
      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
      cluster_labels = kmeans.fit_predict(df_pca_transformed)
      # Calcul du silhouette score pour évaluer la qualité du clustering
      silhouette_avg = silhouette_score(df_pca_transformed, cluster_labels)
      # Affichage du Silhouette score dans Streamlit
      st.write("Silhouette score =", silhouette_avg)
      
      image_path = "//Users/morganemuller/Desktop/DATA/Streamlit/KMeans 3 clusters après ACP.png" 
      st.image(image_path, caption='KMeans 3 clusters', use_column_width=True)
      image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/KMeans 3 clusters avec centroides après ACP.png" 
      st.image(image_path, caption='KMeans 3 clusters avec centroides', use_column_width=True)
      st.wrtite("Analyse des composantes principales")
      st.write("PC1 : influencée par Salaire_net_moyen, Salaire_net_moyen_heure_employé, and Salaire_net_moyen_heure_cadre.")
      st.write("PC2 : influencée par le niveau de diplôme.")
      st.write("PC3 : influencée par Salaire_net_moyen_heure_cadre_moyen et Salaire_net_moyen_heure_ouvrier.")
      st.write("PC4 : influencée par la taille des entreprises.")

      
       