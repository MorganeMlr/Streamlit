#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:40:50 2024

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
from PIL import Image

# Import the data
df=pd.read_csv("df.csv")

st.title("French Industry Project")

###SIDEBAR ###
# Add the project logo 
logo_url = "Images/Logo Datascientest.png"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Le contexte", "Les objectifs", "Les données", "L’exploration et les analyses statistiques", "La modélisation", "L'interprétation", "La conclusion"]
page=st.sidebar.radio("Sélectionnez une partie :", pages)

# Add a line before the section
st.sidebar.markdown("---") 

# Adding logo, names, and LinkedIn links with photos
st.sidebar.markdown("## Project Team")


# Add team members
team_members = [
    {"name": "Morgane MULLER", "photo_url": "Images/Morgane.jpeg", "linkedin_url": "https://www.linkedin.com/in/morgane-muller/"},
    {"name": "Nolwenn RUMEUR", "photo_url": "Images/Nolwenn.jpeg", "linkedin_url": "https://www.linkedin.com/in/nolwennrumeur/"},
    {"name": "Leila BENAMEUR", "photo_url": "Images/Leila.jpeg", "linkedin_url": "https://www.linkedin.com/in/le%C3%AFla-benameur-45292325/"},
    {"name": "Samir BELFEROUM", "photo_url": "Images/Samir.jpeg", "linkedin_url": "https://www.linkedin.com/in/sbelferoum/"},
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
    st.markdown("<h3 style='color: #5930F2;'>Le contexte</h3>", unsafe_allow_html=True)
    st.markdown("Notre projet d’analyse de données porte sur l’étude des inégalités de salaire en France, un sujet vaste et sur lequel de nombreux organismes et chercheurs travaillent déjà actuellement.")
    st.markdown("<u>Plusieurs axes d’études nous ont été proposés :</u>", unsafe_allow_html=True)
    st.write("**Inégalités entre les entreprises en fonction de leur localisation et / ou de leur taille ;**")
    st.write("**Inégalités au sein de la population en fonction du salaire et de la localisation ;**")
    st.write("**Inégalités au sein d’une grande ville française en particulier.**")
              
    st.write("Nous avons choisi de nous concentrer sur les inégalités de salaire sur le territoire français en posant la problématique suivante :")
              
    st.markdown("<h5 style='color: #27DCE0;'>Dans quelle mesure la géographie influence-t-elle les inégalités de salaires en France ?</h5>", unsafe_allow_html=True)
    
    st.write("**La géographie**, dans le cadre de cette problématique, concerne l'étude des facteurs spatiaux et régionaux qui influencent les conditions économiques et sociales des populations et des entreprises. Elle examine comment les caractéristiques physiques, les infrastructures, et les dynamiques locales, notamment économiques, impactent le développement et les niveaux de salaire.")

    st.write("**Les inégalités de salaires**, elles, se réfèrent aux différences de rémunération perçues par les travailleurs en fonction de divers facteurs tels que l'expérience, l'éducation, le secteur d'activité, le genre, et, dans ce cadre, la localisation géographique. Ces disparités peuvent se manifester à travers des écarts de salaire brut, des primes, et des avantages sociaux.")
              
    st.write("Ce projet est réalisé dans le cadre de la formation Data Analyst de DataScientest et ne s’insère pas directement dans les métiers respectifs des membres du groupe. Il s’agit pour nous d’une première, et nous sommes enthousiastes à l’idée de découvrir de près le métier de data analyst. En revanche, ce projet pourrait facilement s’insérer dans le métier d’un data analyst travaillant pour une institution publique (par exemple l’Observatoire des inégalités, une collectivité territoriale voulant en connaître plus sur les inégalités de salaire dans sa localité, ou encore un ministère - à titre d’exemple le ministère de la Cohésion des territoires ou le ministère du Travail)  souhaitant proposer des politiques publiques pour réduire ces inégalités.")

    st.write("• **Du point de vue technique**, les données sont en effet collectées et traitées dans un premier temps par l’INSEE, avant d’être analysées statistiquement et modélisées / visualisées par des agents de cet institut ou d’un autre organisme public.") 
    st.write("• **Du point de vue économique**, les organismes publics mentionnées ci-dessus ont besoin de réaliser des analyses des impacts économiques de ces inégalités afin de proposer de nouvelles politiques publiques visant à réduire ces dernières avant enfin d’évaluer le résultat de ces mêmes politiques publiques.")
    st.write("• **Du point de vue scientifique**, il serait intéressant de réaliser une revue de littérature car de très nombreuses études et recherches existent déjà sur le sujet. Cela nous permettrait de nous appuyer sur des travaux existants pour affiner nos recherches et nos hypothèses. A l’issue de nos travaux, il serait envisageable de diffuser nos résultats à la communauté scientifique et au grand public, via un rapport ou un article.")
                  
## Les objectifs Page 
if page == pages[1]:
    st.markdown("<h3 style='color: ##5930F2;'>Les objectifs</h3>", unsafe_allow_html=True)
    
    st.markdown('<u>Les principaux objectifs que nous nous sommes fixés sont les suivants :</u>', unsafe_allow_html=True)
    st.markdown('<span style="color: #27DCE0; font-weight: bold;">- Identifier les disparités salariales sur le territoire français (métropolitain hors Corse):</span> cartographier les inégalités de salaires horaires en France à l’échelle locale (communes), départementale et régionale', unsafe_allow_html=True)
    st.markdown('<span style="color: #27DCE0; font-weight: bold;">- Analyser les facteurs déterminants de ces inégalités :</span> déterminer les principaux facteurs géographiques, économiques, sociaux et individuels qui influencent les variations des salaires horaires, tels que le niveau de diplôme, les infrastructures de transport telles que les gares et les aéroports, la présence d’entreprises de taille variée, etc.', unsafe_allow_html=True)
    st.markdown('<span style="color: #27DCE0; font-weight: bold;">- Faire des propositions de politiques publiques</span> pouvant participer à la réduction de ces inégalités au vu des conclusions de notre étude et des pistes tirées de notre revue de littérature.', unsafe_allow_html=True)
   
## Les données Page 
if page == pages[2]:
    st.markdown("<h3 style='color: ##5930F2;'>Les données</h3>", unsafe_allow_html=True)
    st.write("Afin d’atteindre les objectifs de notre projet, nous avons utilisé plusieurs jeux de données. De façon générale, ces derniers sont issus de l’INSEE ou de sites gouvernementaux. L’INSEE (Institut National de la Statistique et des Etudes Economiques) est l'institut officiel français qui collecte des données de tous types sur le territoire français. Les données que ce dernier collecte et analyse sont disponibles librement, afin de permettre à tous les organismes publics, citoyens et entreprises privées d’accéder à des informations et analyses intéressantes qui peuvent les concerner.")

    st.markdown('<u>Les jeux de données utilisés sont les suivants :</u>', unsafe_allow_html=True)
    
    st.write("-**'base_etablissement_par_tranche_effectif.csv'** : informations sur le nombre d'entreprises dans chaque ville française classées par taille .")
    st.write("-**'name_geographic_information'**: données géographiques sur les villes françaises (principalement la latitude et la longitude, mais aussi les codes et les noms des régions/départements).")
    st.write("-**'net_salary_per_town_categories'** : salaires par villes française par catégories d'emploi, âge et sexe. Ce fichier fournit des données sur les salaires nets horaires. Les indicateurs sont ventilés selon le sexe, l'âge, la catégorie socioprofessionnelle (hors agriculture).")
    st.write("-**'referentiel-gares-voyageurs.csv'** : informations sur l’intégralité des gares et leur type (TGV, TER, etc.) sur le territoire.")
    st.write("-**'aeroports_france'** : informations sur les aéroports recensés sur le territoire français en 2014.")
    st.write("-**'base-cc-diplomes-formation-2020.csv'** : données de l’INSEE concernant le niveau de diplôme dans les communes françaises en 2020.")
    st.write("-**'prixm2.csv'** : informations donnant le prix moyen au mètre carré des logements dans les communes françaises  en 2014")
    st.write("-**'coordonnees_geo.csv'** : données qui regroupent les coordonnées géographiques (latitude et longitude) des communes françaises")
    
    st.write("Nous disposions d’un jeu de données supplémentaire 'population.csv' : qui contient des informations démographiques par ville, âge, sexe et mode de vie mais nous avons choisi de ne pas l’utiliser. Nous souhaitions nous concentrer sur les communes plutôt que sur les individus comme unité de compte (ce qui permet également d’avoir un jeu de données avec un nombre de lignes relativement restreint).")
    
    st.write("Une fois fusionnés dans un df_final.csv à partir duquel nous travaillerons, nous obtenons un jeu de données avec 4662 lignes et 48 colonnes.")
    
    st.markdown("<h3 style='color: #5930F2;'>Preprocessing</h3>", unsafe_allow_html=True)

    st.write("#### Afficher les données")
    st.dataframe(df.head())
    
    st.markdown('<u>Description des principales colonnes du dataframe :</u>', unsafe_allow_html=True)
 

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
    
    st.write("Afin d’arriver à un modèle performant, il est essentiel de procéder à un prétraitement des données, en vue par exemple de traiter des difficultés comme des informations incomplètes, des valeurs manquantes ou erronées ou encore des bruits parasites liés à l’acquisition de la donnée. Ce prétraitement des données présente plusieurs étapes, que nous allons détailler dans le cadre spécifique de notre projet de clustering.")
    
    st.markdown('<u>Les différentes étapes :</u>', unsafe_allow_html=True)
    
    st.write("**1. Data cleaning**. Pour cette première étape, nous avons d’abord supprimé un certain nombre de colonnes inutiles après la fusion des différents fichiers. Nous avons par ailleurs vérifié s’il y avait des doublons et des valeurs manquantes. Il y en avait assez peu et lorsque c’était le cas nous avons pu remplir ces valeurs manquantes par des données cohérentes. Par exemple, lors de l’absence de prix moyen au mètre carré dans certaines communes, nous avons remplacé la valeur manquante par la moyenne régionale. Nous avons également simplifié le nom des colonnes pour faciliter le travail sur le jeu de données ainsi que la lecture des graphiques lors de l’étape de visualisation des données.")
    st.write("**2. Data transformation**. Ensuite, nous avons eu recours à une normalisation des données afin de ramener les données numériques à une échelle plus petite (ici, entre -1 et 1), mais de façon ponctuelle lorsque cela était requis par l’analyse. Cela a notamment été le cas avant l’utilisation des différents algorithmes de clustering, mais pas pour les graphiques de visualisation des données, pour lesquels nous souhaitions conserver les valeurs brutes pour en faciliter la compréhension.")
    st.write("**3. Data reduction**. De même, l’étape de réduction de la dimension des données a été réalisée de façon ad hoc, principalement via une analyse par composantes principales (PCA) avant l’application d’algorithmes de clustering comme celui des K-Means, avec pour objectif d’améliorer la performance du modèle en facilitant l’analyse des données par l’algorithme.")
    st.write("**4. Data integration**. Comme expliqué précédemment, notre jeu de données final est composé de données variées, que nous avons traitées et regroupées afin de pouvoir en tirer une analyse pertinente par rapport à notre problématique concernant les inégalités de salaires en France métropolitaine.")
    
## L’exploration et les analyses statistiques Page ##
if page == pages[3] : 
    st.markdown("<h3 style='color: #5930F2;'>L'exploration et les analyses statistiques</h3>", unsafe_allow_html=True)
    
    st.write("Avant de nous concentrer sur la modélisation via des algorithmes de clustering, nous avons commencé par réaliser des explorations statistiques sur nos données avec pour objectif d’y découvrir des tendances, caractéristiques et corrélations à examiner plus en profondeur par la suite.") 
    st.markdown('<u>Voici les méthodes auxquelles nous avons eu recours :</u>', unsafe_allow_html=True)
    st.write("**- La visualisation des données**") 
    st.write("**- La matrice de corrélation**")
    st.write("**- Le test ANOVA (analyse de variance**")
    st.write("**- La régression linéaire**")
    
    choix = [
        'La visualisation des données',
        'La matrice de corrélation',
        'Le test ANOVA (analyse de variance)', 
        'La régression linéaire', 
        ]
    
    # Interface utilisateur
    choix = ['La visualisation des données', 'La matrice de corrélation', 'Le test ANOVA (analyse de variance)', 'La régression linéaire']  # Ajoutez d'autres options si nécessaire
    option = st.selectbox("Choix", choix)
    st.write('Le test statistique choisi est :', option)
    
    if option == 'La visualisation des données':
       # Afficher la première image
                  image_path = "Images/Distribution des salaires par régions.png"
                  st.image(image, caption='Distribution des salaires par région', use_column_width=True)
    
    if option == 'La matrice de corrélation':
       # Afficher l'image 
       image_path = "Images/Matrice de corrélation.png"  
       st.image(image_path, caption='La matrice de corrélation', use_column_width=True)
       st.write("La matrice de corrélation, nous permet d’identifier des variables qui sont potentiellement corrélées (plus ou moins fortement) et d’opérer le cas échéant une sélection parmi ces dernières pour le clustering. Ici, l’interprétation de la matrice de corrélation suggère que :")
       st.write("**- Relation entre le niveau de salaire et le prix de l’immobilier.** Le niveau des salaires est un indicateur clé pour expliquer la variation du prix de l'immobilier. En effet, il apparaît que le prix moyen au mètre carré est fortement lié aux différentes catégories de salaires horaires, notamment ceux des femmes et des hommes âgés de 26 à 50 ans. En revanche, le niveau de diplôme apparaît moins significativement corrélé au prix au mètre carré qu'au niveau de salaire. Il existerait par ailleurs **une disparité de genre dans l'impact des salaires sur le prix moyen au m².** En effet, les corrélations entre le salaire horaire moyen des femmes avec le prix au m² sont légèrement plus élevées que celles des hommes dans les mêmes tranches d'âge. Cela pourrait indiquer que l'augmentation des salaires féminins dans une région est particulièrement associée à la hausse du prix de l’immobilier. Cela peut être le signe d'une participation croissante des femmes au marché du travail ou leur insertion à des postes mieux rémunérés.")
       st.write("**- Relation entre le niveau de diplôme et le prix moyen au m².** Le niveau de diplôme Bac+2 est le niveau d'éducation qui a la plus forte corrélation avec le prix moyen au m². Cela suggère que les communes avec une plus grande proportion de personnes ayant un diplôme Bac+2 tendent à avoir des prix immobiliers plus élevés. Au-delà de ce niveau de diplôme, la corrélation diminue, tout comme au-dessous. Cette information suggère que les niveaux de diplômes les plus faibles auraient tendance à s’adapter au prix de l’immobilier - peut-être en habitant hors des zones où la plupart des habitants ont au moins un Bac+2.")
       st.write("**- Relation entre les entreprises et les salaires.** Les corrélations entre total_entreprise, Micro_entreprise, Petite_entreprise, Moyenne_entreprise, et Grande_entreprise avec les différentes catégories de salaires sont en général faibles, indiquant qu'il n'y a pas de relation directe significative entre le nombre d'entreprises et les niveaux de salaires des différentes catégories de travailleurs.")
       st.write("**- Relation entre la population totale et les entreprises.** Il existe une corrélation modérément positive entre total_population et total_entreprise, ce qui est attendu car une population plus importante peut impliquer la présence de plus d'entreprises.")
       st.write("**- Relation entre le niveau de diplôme et les salaires**. Les variables liées au niveau d'éducation (comme No_diplome, BEPC_brevet_DNB, CAP_BEP, Bac2, etc.) montrent des corrélations variables avec les salaires, avec des corrélations plus faibles avec les bas niveaux d'éducation. Cela reflète la tendance générale où des niveaux de diplômes plus élevés sont associés à des salaires plus élevés, bien que la corrélation ne soit pas très forte dans certains cas.")
        
    if option == 'Le test ANOVA (analyse de variance)':
       # Filtrer les colonnes nécessaires pour l'analyse
       df_region_price = df[['nom_région', 'Prixm2Moyen']].dropna()
       # Obtenir les groupes de prix par région
       regions = df_region_price['nom_région'].unique()
       grouped_data = [df_region_price[df_region_price['nom_région'] == region]['Prixm2Moyen'] for region in regions]
       # Effectuer une ANOVA (analyse de la variance) pour comparer les moyennes entre les régions
       anova_result = stats.f_oneway(*grouped_data)
       # Afficher le résultat du test ANOVA avec Streamlit
       st.write(f"**Statistique F : {anova_result.statistic}**")
       st.write(f"**P-value : {anova_result.pvalue}**")
       st.write("Les différences de prix au m² entre les régions sont statistiquement significatives.")
       
       st.markdown('<u>**Interprétation :**</u>', unsafe_allow_html=True)
       st.write("Statistique F élevée (190.91) : Cela signifie qu'il y a une grande différence entre les moyennes des prix au m² dans les différentes régions par rapport à la variance à l'intérieur de chaque région. Plus la statistique F est grande, plus la différence entre les groupes est importante.")
       st.write("P-value de 0.0 : Une p-value de 0.0 signifie que la probabilité d'observer ces différences par hasard est extrêmement faible. Cela nous permet de rejeter l'hypothèse nulle, qui stipule qu'il n'y a pas de différence entre les prix au m² des différentes régions.")

       st.markdown('<u>**Conclusion :**</u>', unsafe_allow_html=True)
       st.write("Les différences entre les prix moyens au m² des différentes régions sont statistiquement significatives. Autrement dit, le prix de l'immobilier varie de manière significative d'une région à l'autre. Nous pouvons en conclure que les régions n'ont pas des prix immobiliers homogènes et que certains facteurs régionaux influencent les prix au m².")
        
    if option == 'La régression linéaire':
       # Sélection des colonnes pertinentes pour la régression
       regression_columns = [
       'total_entreprise', 'Prixm2Moyen', 'total_population'
        ]
       st.markdown('<u>**Conclusion :**</u>', unsafe_allow_html=True)
       st.write("**L'analyse des résultats de la régression OLS montre R² très élevé (0.999)**, ce qui indique que le modèle explique presque toute la variabilité de la variable dépendante, total_entreprise. Les qualifications éducatives (comme BEPC_brevet_DNB, Bac_Brevet_pro, et Bac5) ont un impact positif significatif, tandis que l'absence de diplôme (No_diplome) a un effet négatif. Cependant, la présence d'une forte multicolinéarité soulève des préoccupations sur la robustesse des coefficients individuels, rendant difficile l'évaluation précise de chaque variable. Certaines variables, telles que total_population et Moyenne_entreprise, montrent également des coefficients négatifs, indiquant qu'une augmentation de ces facteurs peut réduire la valeur totale des entreprises.")
       st.write("En résumé, bien que le modèle ait une bonne capacité explicative, des précautions doivent être prises concernant la multicolinéarité et l'interprétation des coefficients.")

## Modélisation Page ##
if page == pages[4] : 
   st.markdown("<h3 style='color: #5930F2;'>La modélisation</h3>", unsafe_allow_html=True)
   
   st.write("**Classification du problème**")
   st.write("Notre projet d’analyse de données porte sur la comparaison des inégalités de salaire en France et s’apparente à du clustering. En effet, il s’agit d’identifier un certain nombre de clusters regroupant des communes françaises ou des départements / régions ayant des similarités (par exemple en termes de prix au mètre carré, de niveau de diplôme, de nombre ou de taille d’entreprises, etc.) afin de mieux comprendre comment ces variables influent sur les inégalités salariales sur le territoire métropolitain.")
   st.write("Le clustering que nous réalisons pour identifier des groupes de communes françaises s’apparente à une tâche de machine learning non supervisée, spécifiquement à la détection de motifs ou à la segmentation. C’est un exemple d’apprentissage non supervisé où l’on cherche à regrouper des entités sans avoir d’informations préalables sur leur classification.")
   st.write("La métrique de performance principale utilisée pour comparer nos modèles est le Silhouette Score. Ce dernier mesure à quel point un point est proche des points de son propre cluster par rapport aux points des autres clusters. Nous l’avons choisi car il indique une valeur par cluster (nous avons un cluster incluant une ville unique, ce qui a tendance à tirer l’index de performance général par le bas). De plus, allant de -1 (mauvais clustering) à 1 (bon clustering), son résultat est assez simple à lire et comprendre (contrairement par exemple à des indices comme Calinsky-Harabasz dont les valeurs peuvent tendre jusqu’à plus l’infini.")

   st.write("**Choix du modèle**")
   st.write("Nous avons fait appel à différents algorithmes de clustering, qui sont :")
   st.write("L’algorithme des K-Means (qui répartit les données en n clusters en minimisant l'inertie (somme des distances entre les points et leurs centres de cluster). Afin d’obtenir le nombre adéquat de clusters nous avons auparavant fait appel à la méthode du coude et à la méthode du dendrogramme. La première permet de lire le nombre idéal de clusters en fonction de la pente de la courbe représentée. L’abscisse du point auquel la courbe change de trajectoire (généralement de façon assez légère) correspond au nombre idéal de clusters dans le cadre de la modélisation. Quant au dendrogramme, il montre visuellement les étapes à chaque niveau où les clusters sont fusionnés, et la hauteur à laquelle les fusions se produisent représente la distance entre les clusters fusionnés. Pour déterminer le nombre idéal de clusters, on cherche un grand saut vertical dans le dendrogramme. En effet, il indique qu'à cet endroit, deux clusters relativement éloignés l'un de l'autre ont été fusionnés. Dans tous les cas de modélisation - que nous verrons par la suite, ces deux méthodes nous ont amené à considérer que le nombre optimal de clusters était trois.")
 
   st.write("Nous avons décidé de retenir l’algorithme des K-Means, associé à la méthode de réduction de dimension qu’est l’analyse par composantes principales. En effet, ce choix nous semblait allier efficacité, simplicité et pertinence dans les résultats. Par opposition, avec l’agglomerative clustering, nous obtenions des dendrogrammes très peu lisibles, ce qui ne nous permettait pas une interprétation simple.") 
   st.write("Nous n’avons pas utilisé des techniques d’optimisation de paramètres car notre jeu de données et les méthodes de clustering auxquelles nous avons eu recours ne s’y prêtaient pas. Nous avons en revanche utilisé des méthodes de réduction de dimension, à titre d’exemple l’analyse par composantes principales, qui ont facilité la lecture des résultats de notre modèle de clustering.")

   st.write("**Application de l’algorithme des K-Means**")
   st.write("Nous avons décidé d’appliquer l’algorithme des K-Means pour trois clusters -  le nombre optimal qui ressortait de nos analyses du coude et les dendrogrammes - dans quatre situations afin d’analyser les clusters qui en ressortent.")

   choix = [
        'Jeu de données original sans ACP',
        'Jeu de données original avec ACP',
        'Jeu de données groupé par département sans ACP' ,
        'Jeu de données groupé par département avec ACP'
        ]
   option = st.selectbox("Choix", choix)
   st.write('La modélisation choisie est :', option)
   
   if option == 'Jeu de données original sans ACP':
      st.write("**Normalisation des données**")
      # Liste des colonnes à normaliser
      columns_to_scale = ['total_entreprise', 'Prixm2Moyen', 'total_population']
      # Afficher la liste des colonnes à normaliser
      st.write("Liste des colonnes à normaliser")
      st.write(columns_to_scale)
      # Normaliser les colonnes sélectionnées
      scaler = StandardScaler()
      df_scaled = scaler.fit_transform(df[columns_to_scale])
      # Remettre les colonnes normalisées dans un DataFrame avec les noms originaux
      df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)
      # Afficher les premières lignes pour vérifier
      st.write("**DataFrame normalisé (premières lignes)**")
      st.dataframe(df_scaled.head())
      # Méthode du coude
      st.write("**Méthode du coude**")
      # Afficher l'image 
      image_path = "Images/Méthode du coude avant ACP.png" 
      st.image(image_path, caption="Méthode du coude avant ACP", use_column_width=True)
      st.write("D'après la méthode du coude, le nombre optimal de clusters semble être **3**.")
      # Méthode du dendrogramme 
      st.write("**Méthode du dendrogramme**")
      # Afficher l'image 
      image_path = "Images/ Méthode du dendrogramme avant ACP.png"  
      st.image(image_path, caption="Méthode du dendrogramme avant ACP", use_column_width=True)
      st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être **3**.")
      # Algorithme des K-Means 
      # Affichage des coordonnées des centroïdes des clusters
      # Afficher l'image 
      image_path = "Images/Coordonnées centroides sans ACP.png" 
      st.image(image_path, caption="Coordonnées centroides sans ACP", use_column_width=True)
      image_path = "Images/Clusters avec centroides.png"  
      st.image(image_path, caption="clusters avec centroides", use_column_width=True)
      # Evaluation de la qualité du clustering
      st.write("**Silhouette Score : 0.49770381682419584**")
      st.write("**Taille du cluster 1 : 800**")
      st.write("**Taille du cluster 2 : 3861**")
      st.write("**Taille du cluster 3 : 1**")
      # Afficher l'image 
      image_path = "Images/Silhouette score avant ACP.png" 
      st.image(image_path, caption="Silhouette score avant ACP", use_column_width=True)
      st.write("Nous obtenons un Silhouette score moyen (environ 0,5). Le Silhouette Plot nous montre que les clusters 1 et 2 sont plutôt bien définis, mais que la présence d'une ville unique dans le cluster 3 (Paris) tire vers le bas la valeur du Silhouette Score.")

   if option == 'Jeu de données original avec ACP':
      st.write("**Analyse par composantes principales**")
      # Afficher l'image 
      image_path = "Images/Variance expliquée par composantes principales.png" 
      st.image(image_path, caption='Variance expliquée par composantes principales', use_column_width=True)
      # Méthode du coude
      st.write("**Méthode du coude**")
      # Afficher l'image 
      image_path = "Images/Méthode du coude après ACP.png"  
      st.image(image_path, caption="Méthode du coude après ACP", use_column_width=True)
      st.write("D'après la méthode du coude, le nombre optimal de clusters semble être **3**.")
      # Méthode du dendrogramme 
      st.write("**Méthode du dendrogramme**")
      # Afficher l'image 
      image_path = "Images/Méthode du dendrogramme après ACP.png"  
      st.image(image_path, caption="Méthode du dendrogramme après ACP", use_column_width=True)
      st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être **3**.")
      # Evaluation de la qualité du clustering
      image_path ="Images/KMeans 3 clusters après ACP.png"
      st.image(image_path, caption="KMens 3 clusters après ACP", use_column_width=True)
      st.write("**Silhouette score = 0.5365082268991479**")
      image_path ="Images/K-Means avec 3 clusters et centroides.png"
      st.image(image_path, caption="K-Means avec 3 clusters et centroides", use_column_width=True)
      st.write("**Analyse des composantes principales :**")
      st.write("**PC1 : influencée par Salaire_net_moyen, Salaire_net_moyen_heure_employé, and Salaire_net_moyen_heure_cadre**")
      st.write("**PC2 : influencée par le niveau de diplôme**")
      
   if option == 'Jeu de données groupé par département sans ACP' : 
      st.write("**Normalisation des données**")
       # Liste des colonnes à normaliser
      columns_to_scale = ['total_entreprise','Prixm2Moyen','total_population','Micro_entreprise',
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
      # Afficher la liste des colonnes à normaliser
      st.write("Liste des colonnes à normaliser")
      st.write(columns_to_scale)
      # Normaliser les colonnes sélectionnées
      scaler = StandardScaler()
      df_scaled = scaler.fit_transform(df[columns_to_scale])
      # Remettre les colonnes normalisées dans un DataFrame avec les noms originaux
      df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)
      # Afficher les premières lignes pour vérifier
      st.write("**DataFrame normalisé (premières lignes)**")
      st.dataframe(df_scaled.head())
      # Méthode du coude
      st.write("**Méthode du coude**")
      # Afficher l'image 
      image_path = "Images/Méthode du coude sans ACP DEP.png" 
      st.image(image_path, caption="Méthode du coude sans ACP par départements", use_column_width=True)
      st.write("D'après la méthode du coude, le nombre optimal de clusters semble être **3** ou **4**.")
      # Méthode du dendrogramme 
      st.write("**Méthode du dendrogramme**")
      # Afficher l'image 
      image_path = "Images/Dendrogramme sans ACP DEP.png"  
      st.image(image_path, caption="Méthode du dendrogramme sans ACP par départements", use_column_width=True)
      st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être **3**.")
      # Algorithme des K-Means 
      # Affichage des coordonnées des centroïdes des clusters
      # Afficher l'image 
      image_path = "Images/Coordonnées centroides sans ACP DEP.png" 
      st.image(image_path, caption="Coordonnées centroides sans ACP paar départements", use_column_width=True)
      image_path = "Images/Centroides sans ACP DEP.png" 
      st.image(image_path, caption="Clusters avec centroides sans ACP par départements", use_column_width=True)
      # Evaluation de la qualité du clustering
      st.write("**Silhouette Score : 0.5398242144044401**")
      st.write("**Taille du cluster 1 : 73**")
      st.write("**aTille du cluster 2 : 18**")
      st.write("**Taille du cluster 3 : 3**")
      # Afficher l'image 
      image_path = "Images/Silhouette Score sans ACP DEP.png" 
      st.image(image_path, caption="Silhouette score sans ACP par départements", use_column_width=True)
      st.write("Nous obtenons un Silhouette score correct (environ 0,54), ce qui montre que les clusters sont plutôt bien définis, avec une importante variabilité dans la taille de ces derniers en termes de nombre de département qu'ils contiennent.")
   
   if option == 'Jeu de données groupé par département avec ACP' :
      st.write("**Analyse par composantes principales**")
      # Afficher l'image 
      image_path = "Images/Variance expliquée par composantes DEP.png" 
      st.image(image_path, caption='Variance expliquée par composantes principales par départements', use_column_width=True)
      # Méthode du coude
      st.write("**Méthode du coude**")
      # Afficher l'image 
      image_path = "Images/Méthode du coude avec ACP DEP .png"  
      st.image(image_path, caption="Méthode du coude après ACP", use_column_width=True)
      st.write("D'après la méthode du coude, le nombre optimal de clusters semble être **2** ou **3**.")
      # Méthode du dendrogramme 
      st.write("**Méthode du dendrogramme**")
      # Afficher l'image 
      image_path = "Images/Dendrogramme avec ACP DEP.png"  
      st.image(image_path, caption="Méthode du dendrogramme après ACP", use_column_width=True)
      st.write("D'après la méthode du dendrogramme, le nombre optimal de clusters semble être **3**.")
      # Evaluation de la qualité du clustering
      image_path ="Images/KMeans 3 clusters DEP.png"
      st.image(image_path, caption="KMens 3 clusters après ACP", use_column_width=True)
      st.write("**Silhouette score = 0.5097185102337607**")
      image_path ="Images/KMeans 3 clusters et centroides ACP DEP.png"
      st.image(image_path, caption="K-Means avec 3 clusters et centroides", use_column_width=True)
      st.write("**Analyse des composantes principales :**")
      st.write("**PC1 explique environ 75,5% de la variance totale, tandis PC2 explique environ 10,9% de la variance.**")
      st.write("**PC1 : est influencée majoritairement par les variables suivantes : Salaire_net_moyen_heure_26_50, Salaire_net_moyen_heure_26_50_homme, Salaire_net_moyen, Salaire_net_moyen_heure_26_50_femme, Salaire_net_moyen_homme.**")
      st.write("**C'est à dire qu'elle est surtout influencée par le niveau de salaire.**")
      st.write("**PC2 : est majoritairement influencée par les variables suivantes : CAP_BEP, BEPC_brevet_DNB, Gare, No_diplome, Bac_Brevet_pro.**")
      st.write("**C'est à dire qu'elle est surtout influencée par le niveau de diplôme.**")
      
## Interprétation Page ##
if page == pages[5] : 
      st.markdown("<h3 style='color: #5930F2;'>L'interprétation</h3>", unsafe_allow_html=True)
      st.write("Afin de faciliter la lecture des résultats, c’est à dire les spécificités des clusters obtenus, nous avons décidé d’utiliser deux outils : un tableau statistique avec certaines informations clés sur les clusters en question et un nuage de mots représentant les villes ou départements constituant le cluster, dont la taille varie selon le nombre d’habitants de ces entités géographiques.")
   
      st.write("Que ce soit avec ou sans ACP, les clusters sont similaires : on trouve Paris à part les tandis que les grandes métropoles et les zones plus rurales ou intermédiaires sont regroupées dans les autres clusters.")
      
      choix = [
           'Jeu de données original',
           'Jeu de données groupé par départements',
           ]
      option = st.selectbox("Choix", choix)
      
      if option == 'Jeu de données original' : 
          st.markdown("<h3 style='color: #5930F2;'>Jeu de données original</h3>", unsafe_allow_html=True)
          image_path ="Images/clusters jeu de données original sans ACP .png"
          st.image(image_path, caption="Nuage de mots pour les clusters obtenus à partir du jeu de données original sans ACP", use_column_width=True)
          image_path ="Images/clusters jeu de données original avec ACP.png"
          st.image(image_path, caption="Nuage de mots pour les clusters obtenus à partir du jeu de données original avec ACP", use_column_width=True)
          st.write("Que ce soit avec ou sans ACP, les clusters sont similaires : on trouve Paris à part les tandis que les grandes métropoles et les zones plus rurales ou intermédiaires sont regroupées dans les autres clusters.")
          st.markdown('<u>Voici une interprétation de la formation des clusters à partir du jeu de données original :</u>', unsafe_allow_html=True)
          st.write("**Cluster 0 (Cluster 2 après ACP) : Grandes villes de région parisienne et villes rurales de province** (Exemples : Ambronay, Champigny-sur-Marne, Maisons-Alfort, Clichy, Puteaux, Sartrouville, Arbent, Attignat, Bâgé-le-Châtel, Beynost, Cessy, Chalamont, Châtillon-en-Michaille, Châtillon-sur-Chalaronne, Chevry, Civrieux, Confrançon, Culoz, Divonne-les-Bains, Farges, Feillens, Colombes, Francheleins, Montreuil, Massy, etc.) Ce cluster regroupe des villes avec une population en moyenne plus faible (8700 habitants). Elles apparaissent majoritairement situées en région parisienne pour les plus grandes et dans des régions rurales et moins urbanisées pour les plus petites. Les salaires y sont moyens par rapport au niveau national, tout comme le prix moyen du mètre carré. Les villes de ce cluster présentent en moyenne des niveaux de diplômes légèrement plus faibles que dans les deux autres.")
          st.write("**Cluster 1 (Cluster 0 après ACP) : Grandes et moyennes villes de province**(Exemples : Marseille, Nice, Rennes, Ambérieu-en-Bugey, Bourg-en-Bresse, Oyonnax, Bellegarde-sur-Valserine, Lons-le-Saunier, Montmorot, Saint-Claude, Gex, Thonon-les-Bains, Annemasse, Ville-la-Grand, Ferney-Voltaire, Saint-Genis-Pouilly, Sallanches, Cluses, Bonneville, La Roche-sur-Foron, Tours, Tourcoing, Béziers, etc.). Ce cluster est celui qui réunit le plus grand nombre de villes. Il s’agit, de façon générale, de villes de province. La population moyenne des villes de ce cluster est modérée (environ 10 000 habitants), tirée par le haut par les grandes métropoles. Les habitants de ces villes présentent en moyenne un niveau d'éducation modéré (CAP/BEP, Bac, Bac+2). Les salaires y sont en moyenne plus bas qu’ailleurs, tout comme le prix moyen du mètre carré.")  
          st.write("**Cluster 2 (Cluster 1 après ACP) : Paris constitue à elle seule un cluster**, avec des caractéristiques exceptionnelles : une population très élevée (plus de 2 millions), une très grande concentration d'entreprises, une forte présence d'aéroports et de gares et un haut niveau d'éducation (forte proportion de diplômés Bac+5).")
      
      if option == 'Jeu de données groupé par départements' : 
          st.markdown("<h3 style='color: #5930F2;'>Jeu de données groupé par départements</h3>", unsafe_allow_html=True)
          image_path ="Images/clusters jeu de données groupé par département sans ACP .png"
          st.image(image_path, caption="Nuage de mots pour les clusters obtenus à partir du jeu de données groupé par département sans ACP", use_column_width=True)
          image_path ="Images/clusters jeu de données groupé par département avec ACP .png"
          st.image(image_path, caption="Nuage de mots pour les clusters obtenus à partir du jeu de données original avec ACP", use_column_width=True)
          st.write("On constate que les résultats du clustering avec et sans ACP sont identiques dans le cas du jeu de données groupé par départements, ce qui signifie que la réduction de dimension n’a pas eu d’impact significatif sur la formation des clusters.")
          st.markdown('<u>Voici une interprétation de la formation des clusters à partir du jeu de données groupé par département :"</u>', unsafe_allow_html=True)
          st.write("**Cluster 0 : Départements ruraux ou semi-ruraux** (Ain, Aisne, Allier, Alpes-de-Haute-Provence, Hautes-Alpes, Ardèche, Ardennes, Ariège, Aube, Aude, Aveyron, Calvados, Cantal, Charente, Charente-Maritime, Cher, Corrèze, Côte-d'Or, Côtes-d'Armor, Creuse, Dordogne, Doubs, Drôme, Eure, Eure-et-Loir, Finistère, Gard, Gers, Ille-et-Vilaine, Indre, Indre-et-Loire, Jura, Landes, Loir-et-Cher, Loire, Haute-Loire, Loiret, Lot, Lot-et-Garonne, Lozère, Maine-et-Loire, Manche, Marne, Haute-Marne, Mayenne, Meurthe-et-Moselle, Meuse, Morbihan, Moselle, Nièvre, Orne, Pas-de-Calais, Puy-de-Dôme, Pyrénées-Atlantiques, Hautes-Pyrénées, Pyrénées-Orientales, Haut-Rhin, Haute-Saône, Saône-et-Loire, Sarthe, Savoie, Deux-Sèvres, Somme, Tarn, Tarn-et-Garonne, Vaucluse, Vendée, Vienne, Haute-Vienne, Vosges, Yonne, Territoire de Belfort) Ce cluster regroupe principalement des départements ruraux ou semi-ruraux. Ces départements se distinguent souvent par un taux de diplômés inférieur à la moyenne nationale. Le nombre d’entreprises dans ces départements a également tendance à être plus bas, et l’emploi y est souvent centré sur les secteurs agricoles ou industriels. Le salaire moyen est inférieur dans ces départements par rapport aux zones urbaines et on y trouve moins de grandes entreprises, ce qui indique une économie moins dépendante des grandes villes et plus axée sur des secteurs traditionnels ou régionaux. Enfin, d’un point de vue géographique, un grand nombre de ces départements sont situés dans des régions montagneuses ou à faible densité de population, éloignés des grandes métropoles.")
          st.write("**Cluster 1 : Départements urbains avec un dynamisme économique** (Alpes-Maritimes, Bouches-du-Rhône, Haute-Garonne, Gironde, Hérault, Isère, Loire-Atlantique, Nord, Oise, Bas-Rhin, Rhône, Haute-Savoie, Seine-Maritime, Seine-et-Marne, Var, Essonne, Seine-Saint-Denis, Val-de-Marne, Val-d'Oise) Ce groupe rassemble des départements à tendance plus urbaine, une population relativement plus jeune et un taux de qualification plus élevé. Ces départements comprennent des régions métropolitaines et des villes universitaires comme Toulouse (Haute-Garonne) et Marseille (Bouches-du-Rhône). Le salaire moyen y est souvent plus élevé que dans le premier cluster, et ces départements comptent un plus grand nombre de grandes entreprises. Il y a également un plus grand dynamisme économique, avec des industries modernes telles que les services, les technologies, et le tourisme (Alpes-Maritimes). Beaucoup de ces départements se situent dans des zones littorales (Alpes-Maritimes, Bouches-du-Rhône) ou autour de grandes agglomérations (Haute-Garonne avec Toulouse, Rhône avec Lyon).")  
          st.write("**Cluster 2 : Départements riches et très urbanisés** (Paris, Yvelines, Hauts-de-Seine) Ce dernier cluster regroupe principalement les départements les plus riches et les plus urbanisés de France. Ce sont des zones à très forte densité de population, avec une main-d’œuvre hautement qualifiée. Le salaire moyen est ici bien plus élevé que la moyenne nationale, surtout en raison de la présence de nombreux cadres. Ce sont aussi les départements avec le plus grand nombre de sièges sociaux de grandes entreprises, notamment dans les Hauts-de-Seine (La Défense). Géographiquement, il s’agit de Paris et ses environs immédiats. Ces départements constituent le cœur économique de la France.")

## Conclusion Page ##
if page == pages[6] : 
    st.markdown("<h3 style='color: ##5930F2;'>La conclusion</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #993399;'>Bilan</h3>", unsafe_allow_html=True)
    st.write("**En conclusion, pour répondre à la problématique et évaluer les objectifs que nous nous étions fixés au début du projet, nous pouvons conclure sur les points suivants :**")
    st.write("- La géographie influence les inégalités de salaire dans la mesure où il est possible de regrouper les départements métropolitains et les villes dans trois groupes bien définis répartis inégalement sur le territoire.")
    st.write("- Les facteurs qui influencent principalement ce regroupement sont la dimension urbaine ou rurale de ces zones, qui elle-même influence le dynamisme économique et donc la richesse de ces localités. On remarque notamment la position très spécifique de Paris et de sa proche banlieue par rapport au reste de la France.")
    st.write("- Un point à prendre en compte est que nous nous sommes intéressés aux inégalités de salaire sur le territoire français, mais pas aux inégalités de niveau de vie. En effet, prendre en compte un salaire horaire sans tenir compte du coût de la vie à un endroit spécifique est une démarche intéressante, mais incomplète.")
    st.write("- En termes de préconisations de politiques publiques dans le cadre de la lutte contre les inégalités, il semble intéressant de s’intéresser aux clusters formés de façon différente, notamment en prenant en compte les réalités diverses des zones plus ou moins urbaines et rurales. Contrairement aux idées reçues, pour reprendre les termes du rapport du CGET de 2018, « il n'y a pas, d'un côté, des métropoles dynamiques et, de l'autre, des territoires périphériques sacrifiés sur l'autel de la mondialisation ». Il y a des réalités territoriales diverses entre ceux qui foncent, ceux qui s'accrochent et d'autres qui décrochent. C’est ce que nous constatons dans la formation des clusters, où, par exemple, de nombreuses villes de banlieue parisienne présentent des caractéristiques similaires à des communes rurales.")
    st.write("Cependant, en termes de process métier, si la découverte des méthodes du clustering était très intéressante dans le cadre de ce projet de groupe, avec du recul, nous aurions peut-être choisi un autre sujet car si nous avons développé des compétences intéressantes en termes de clustering, nous pensons que nous avons potentiellement moins de chances de réutiliser ces acquis au cours de notre vie professionnelle future, contrairement à d’autres notions abordées au cours de la formation et sur lesquelles d’autres projets sont basés.")
    
    st.write("Un point positif de ces mois de projet qu’il nous semble intéressant de souligner est que notre groupe a bien fonctionné : même si nous avions peu de connaissances et compétences dans le domaine au début du projet, chacun s’est investi fortement dans le travail de recherche, ce qui nous a permis de mener à bien notre travail. Nous avons apprécié travailler les uns avec les autres et une adéquate division des tâches et la bonne entente qui régnait dans le groupe ont fortement contribué à notre réussite.")
