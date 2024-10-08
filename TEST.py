#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:19:31 2024

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

###SIDEBAR ###
# Add the project logo 
logo_url = "l/Users/morganemuller/Desktop/DATA/Streamlit/Images/logo.png"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Contexte et objectifs du projet", "Revue de litterature", "Les objectifs", "Les données", "L’exploration et les analyses statistiques", "Modélisation", "Analyse des Résultats", "Conclusions & Perspectives"]
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
              Notre projet d’analyse de données porte sur l’étude des inégalités de salaire en France, un sujet vaste et sur lequel de nombreux organismes et chercheurs travaillent déjà actuellement. Plusieurs axes d’études nous ont été proposés :
              - Inégalités entre les entreprises en fonction de leur localisation et / ou de leur taille ;
              - Inégalités au sein de la population en fonction du salaire et de la localisation ;
              - Inégalités au sein d’une grande ville française en particulier.
              
              Nous avons choisi de nous concentrer sur les inégalités de salaire sur le territoire français en posant la problématique suivante :
              """)
              
    st.markdown("<h1 style='color: violet; Dans quelle mesure la géographie influence-t-elle les inégalités de salaires en France ?", unsafe_allow_html=True)
    
    st.write("**La géographie**, dans le cadre de cette problématique, concerne l'étude des facteurs spatiaux et régionaux qui influencent les conditions économiques et sociales des populations et des entreprises. Elle examine comment les caractéristiques physiques, les infrastructures, et les dynamiques locales, notamment économiques, impactent le développement et les niveaux de salaire.")

    st.write("**Les inégalités de salaires**, elles, se réfèrent aux différences de rémunération perçues par les travailleurs en fonction de divers facteurs tels que l'expérience, l'éducation, le secteur d'activité, le genre, et, dans ce cadre, la localisation géographique. Ces disparités peuvent se manifester à travers des écarts de salaire brut, des primes, et des avantages sociaux.")
              
    st.write("Ce projet est réalisé dans le cadre de la formation Data Analyst de DataScientest et ne s’insère pas directement dans les métiers respectifs des membres du groupe. Il s’agit pour nous d’une première, et nous sommes enthousiastes à l’idée de découvrir de près le métier de data analyst. En revanche, ce projet pourrait facilement s’insérer dans le métier d’un data analyst travaillant pour une institution publique (par exemple l’Observatoire des inégalités, une collectivité territoriale voulant en connaître plus sur les inégalités de salaire dans sa localité, ou encore un ministère - à titre d’exemple le ministère de la Cohésion des territoires ou le ministère du Travail)  souhaitant proposer des politiques publiques pour réduire ces inégalités.")

    st.write("**Du point de vue technique**, les données sont en effet collectées et traitées dans un premier temps par l’INSEE, avant d’être analysées statistiquement et modélisées / visualisées par des agents de cet institut ou d’un autre organisme public.") 
    st.write("**Du point de vue économique**, les organismes publics mentionnées ci-dessus ont besoin de réaliser des analyses des impacts économiques de ces inégalités afin de proposer de nouvelles politiques publiques visant à réduire ces dernières avant enfin d’évaluer le résultat de ces mêmes politiques publiques.")
    st.write("**Du point de vue scientifique**, il serait intéressant de réaliser une revue de littérature car de très nombreuses études et recherches existent déjà sur le sujet. Cela nous permettrait de nous appuyer sur des travaux existants pour affiner nos recherches et nos hypothèses. A l’issue de nos travaux, il serait envisageable de diffuser nos résultats à la communauté scientifique et au grand public, via un rapport ou un article.")
                  
## Le Jeu De Données Page 
if page == pages[1]:
    st.write("### Revue de litterature")
    
    st.write("Les inégalités de salaire en France métropolitaine sont un phénomène complexe influencé par des facteurs géographiques, socio-économiques et structurels. Le recours à des méthodes quantitatives comme le clustering permet d'identifier des schémas sous-jacents dans la distribution des revenus et des disparités salariales entre différentes régions et catégories de travailleurs.")
    
    st.write("**Clustering et géographie des salaires en France**")
    st.write("Les techniques de **clustering** permettent de diviser le territoire en zones homogènes sur la base de critères économiques et sociaux. Selon l’étude de Duranton et Puga (2018), ces techniques permettent d'identifier des pôles de richesse et de pauvreté, et de mettre en lumière l'existence d'inégalités salariales significatives entre les zones urbaines et rurales. En effet, les régions métropolitaines et les grandes villes présentent des niveaux de salaire bien supérieurs à ceux observés dans les zones périphériques et rurales. Ces écarts salariaux sont principalement attribués à la concentration des entreprises de haute technologie et des industries à forte valeur ajoutée dans les centres urbains (Duranton & Puga, 2018).")
    st.write("**Polarisation du marché du travail**")
    st.write("Le marché du travail en France est marqué par une polarisation croissante, avec une concentration accrue des emplois à haute et basse qualification, généralement autour des grandes villes, tandis que les emplois intermédiaires tendent à disparaître. Cette évolution a des répercussions directes sur la distribution des salaires, entraînant une augmentation des inégalités. Le **clustering** a permis à plusieurs chercheurs d’identifier trois principaux groupes de travailleurs : ceux occupant des postes très qualifiés et bien rémunérés, ceux dans des emplois peu qualifiés et mal rémunérés, et enfin une classe intermédiaire en déclin (Autor, 2019).")
    st.write("**Facteurs explicatifs des inégalités salariales**")
    st.write("Les inégalités de salaire en France métropolitaine ne s'expliquent pas seulement par des facteurs géographiques, mais également par des caractéristiques individuelles telles que le genre, l'âge et le niveau de diplôme. Le recours au **clustering** permet d’analyser les interactions complexes entre ces différents facteurs. Par exemple, une étude menée par Meurs, Pailhé et Simon (2016) montre que les femmes, les travailleurs issus de l'immigration et les jeunes sont davantage susceptibles d'occuper des emplois précaires et moins bien rémunérés. Même si nous ne nous intéresserons pas à ces critères individuels dans notre étude, il est intéressant de voir que ces résultats mettent en lumière l'importance de prendre en compte des facteurs socio-démographiques dans l'analyse des inégalités de salaire.")
    st.write("**Politiques publiques et réduction des inégalités**")
    st.write("Les résultats obtenus grâce aux méthodes de **clustering** sont également utilisés pour évaluer l'efficacité des politiques publiques visant à réduire les inégalités de salaire. Les analyses spatiales permettent de cibler les zones les plus affectées par les écarts salariaux et de mettre en place des programmes adaptés aux spécificités régionales. Selon Boulant, Brezzi et Veneri (2020), les politiques de développement régional, telles que la décentralisation des activités économiques et l'amélioration de l'accès à l'éducation et à la formation professionnelle, sont cruciales pour réduire les inégalités de salaire dans les territoires les plus défavorisés. En somme, les techniques de **clustering** offrent une approche adéquate pour comprendre et analyser les inégalités de salaire en France métropolitaine. Ces méthodes permettent de révéler des dynamiques complexes, notamment la polarisation géographique et sociale du marché du travail, ainsi que l'impact des caractéristiques individuelles et des politiques publiques sur la répartition des revenus. Toutefois, il est essentiel de continuer à affiner ces analyses pour mieux comprendre les mécanismes à l'origine de ces inégalités et proposer des solutions adaptées. C’est pourquoi nous avons décidé de mener ce projet de recherche en nous concernant sur l’échelle locale (communes et départements / régions), tout en intégrant des données que nous trouvions peu mises en avant dans les études déjà réalisées (à titre d’exemple, la présence d’infrastructures de transports telles que les gares et les aéroports ou des données sur le -s prix de l’immobilier).") 

## Les objectifs Page 
    if page == pages[2]:
        st.write("### Les objectifs")
    
        st.write("Les principaux objectifs que nous nous sommes fixés sont les suivants : ")   
        st.write("**- Identifier les disparités salariales sur le territoire français (métropolitain hors Corse) :** cartographier les inégalités de salaires horaires en France à l’échelle locale (communes), départementale et régionale")
        st.write("**-Analyser les facteurs déterminants de ces inégalités :** déterminer les principaux facteurs géographiques, économiques, sociaux et individuels qui influencent les variations des salaires horaires, tels que le niveau de diplôme, les infrastructures de transport telles que  les gares et les aéroports, la présence d’entreprises de taille variée, etc.")
        st.write("-**Faire des propositions de politiques publiques** pouvant participer à la réduction de ces inégalités au vu des conclusions de notre étude et des pistes tirées de notre revue de littérature.")
    
        st.write("Les différents membres du groupe ne sont pas experts des problématiques liées aux inégalités de salaire en France, mais sont curieux d’en apprendre plus sur ce sujet qui concerne directement tous les citoyens. Des projets similaires sont déjà menés par différents organismes publics et les principaux facteurs des inégalités de salaires en France sont déjà connus. Nous ne prétendons pas découvrir de nouveaux facteurs d’inégalités, mais bien comprendre de l’intérieur comment des data analysts ont pu trouver les réponses à ces questions cruciales pour les politiques publiques en France, et pourquoi pas réfléchir à des idées de politiques publiques de réduction des inégalités qui pourraient par la suite s’insérer dans les métiers respectifs des membres de l’équipe.")
        st.write("Afin de mieux comprendre les données sur lesquelles nous allions travailler, nous avons contacté l’INSEE dès le début de notre projet, dans un premier temps par téléphone. Suite à un échange avec une conseillère, nous avons été invités à faire une demande par mail (contact@contact-insee.fr - Référence de la demande : [1806173-1718020977]). Notre demande portait sur la possibilité d’obtenir des données plus actuelles (celles dont nous disposons datent de 2014) et potentiellement des fichiers additionnels en rapport avec notre problématique. Il nous a été répondu que l’INSEE ne disposait pas d'éléments permettant de répondre à notre demande mais qu’un service en charge de ces questions avait été sollicité. Une relance a été effectuée en date du 20/07/24, sans réponse pour le moment.")
   
## Les données Page 
    if page == pages[3]:
        st.write("### Les données")
    st.write("Afin d’atteindre les objectifs de notre projet, nous avons utilisé plusieurs jeux de données. De façon générale, ces derniers sont issus de l’INSEE ou de sites gouvernementaux. L’INSEE (Institut National de la Statistique et des Etudes Economiques) est l'institut officiel français qui collecte des données de tous types sur le territoire français. Les données que ce dernier collecte et analyse sont disponibles librement, afin de permettre à tous les organismes publics, citoyens et entreprises privées d’accéder à des informations et analyses intéressantes qui peuvent les concerner.")
    
    st.markdown("""
    ### Compréhension et manipulation des données
    """)

    st.write("Les jeux de données utilisés sont les suivants :")
    
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
    
    st.markdown("### Preprocessing")

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
    
    st.write("Afin d’arriver à un modèle performant, il est essentiel de procéder à un prétraitement des données, en vue par exemple de traiter des difficultés comme des informations incomplètes, des valeurs manquantes ou erronées ou encore des bruits parasites liés à l’acquisition de la donnée. Ce prétraitement des données présente plusieurs étapes, que nous allons détailler dans le cadre spécifique de notre projet de clustering.")
    
    st.write("**1. Data cleaning**. Pour cette première étape, nous avons d’abord supprimé un certain nombre de colonnes inutiles après la fusion des différents fichiers. Nous avons par ailleurs vérifié s’il y avait des doublons et des valeurs manquantes. Il y en avait assez peu et lorsque c’était le cas nous avons pu remplir ces valeurs manquantes par des données cohérentes. Par exemple, lors de l’absence de prix moyen au mètre carré dans certaines communes, nous avons remplacé la valeur manquante par la moyenne régionale. Nous avons également simplifié le nom des colonnes pour faciliter le travail sur le jeu de données ainsi que la lecture des graphiques lors de l’étape de visualisation des données.")
    st.write("**2. Data transformation**. Ensuite, nous avons eu recours à une normalisation des données afin de ramener les données numériques à une échelle plus petite (ici, entre -1 et 1), mais de façon ponctuelle lorsque cela était requis par l’analyse. Cela a notamment été le cas avant l’utilisation des différents algorithmes de clustering, mais pas pour les graphiques de visualisation des données, pour lesquels nous souhaitions conserver les valeurs brutes pour en faciliter la compréhension.")
    st.write("**3. Data reduction**. De même, l’étape de réduction de la dimension des données a été réalisée de façon ad hoc, principalement via une analyse par composantes principales (PCA) avant l’application d’algorithmes de clustering comme celui des K-Means, avec pour objectif d’améliorer la performance du modèle en facilitant l’analyse des données par l’algorithme.")
    st.write("**4. Data integration**. Comme expliqué précédemment, notre jeu de données final est composé de données variées, que nous avons traitées et regroupées afin de pouvoir en tirer une analyse pertinente par rapport à notre problématique concernant les inégalités de salaires en France métropolitaine.")
    
## L’exploration et les analyses statistiques Page ##
if page == pages[2] : 
    st.write("### L’exploration et les analyses statistiques")
    
    st.write("Avant de nous concentrer sur la modélisation via des algorithmes de clustering, nous avons commencé par réaliser des explorations statistiques sur nos données avec pour objectif d’y découvrir des tendances, caractéristiques et corrélations à examiner plus en profondeur par la suite.") 
    st.write("Voici les méthodes auxquelles nous avons eu recours : ")
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
        # Afficher première image 
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Distribution des salaires par régions.png"  
        st.image(image_path, caption='Distribution des salaires par région', use_column_width=True)
        # Afficher la deuxième image
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Entreprises par région.png"  
        st.image(image_path, caption='Entreprises par région', use_column_width=True)
        # Afficher la troisième image 
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Ecart-type des salaires par région.png"  
        st.image(image_path, caption='Ecart-type des salaires par région', use_column_width=True)
        # Afficher la quatrième image 
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Evolution des salaires en fonction de l'âge.png"  
        st.image(image_path, caption='Evolution des salaires en fonction de l âge', use_column_width=True)
        # Afficher la cinquième image 
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Carte du prix moyen de l'immobilier par département.png"  
        st.image(image_path, caption='Carte du prix moyen de l immobilier par département', use_column_width=True)
        # Afficher la cinquième image 
        image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Carte du salaire net moyen par département.png"  
        st.image(image_path, caption='Carte du salaire net moyen par département', use_column_width=True)
        
        
    if option == 'La matrice de corrélation':
       # Afficher l'image 
       image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Images/Matrice de corrélation.png"  
       st.image(image_path, caption='La matrice de corrélation', use_column_width=True)
       st.write("La matrice de corrélation, nous permet d’identifier des variables qui sont potentiellement corrélées (plus ou moins fortement) et d’opérer le cas échéant une sélection parmi ces dernières pour le clustering. Ici, l’interprétation de la matrice de corrélation suggère que :")
       st.write("**- Relation entre le niveau de salaire et le prix de l’immobilier.** Le niveau des salaires est un indicateur clé pour expliquer la variation du prix de l'immobilier. En effet, il apparaît que le prix moyen au mètre carré est fortement lié aux différentes catégories de salaires horaires, notamment ceux des femmes et des hommes âgés de 26 à 50 ans. En revanche, le niveau de diplôme apparaît moins significativement corrélé au prix au mètre carré qu'au niveau de salaire. Il existerait par ailleurs **une disparité de genre dans l'impact des salaires sur le prix moyen au m².** En effet, les corrélations entre le salaire horaire moyen des femmes avec le prix au m² sont légèrement plus élevées que celles des hommes dans les mêmes tranches d'âge. Cela pourrait indiquer que l'augmentation des salaires féminins dans une région est particulièrement associée à la hausse du prix de l’immobilier. Cela peut être le signe d'une participation croissante des femmes au marché du travail ou leur insertion à des postes mieux rémunérés.")
       st.write("**- Relation entre le niveau de diplôme et le prix moyen au m².** Le niveau de diplôme Bac+2 est le niveau d'éducation qui a la plus forte corrélation avec le prix moyen au m². Cela suggère que les communes avec une plus grande proportion de personnes ayant un diplôme Bac+2 tendent à avoir des prix immobiliers plus élevés. Au-delà de ce niveau de diplôme, la corrélation diminue, tout comme au-dessous. Cette information suggère que les niveaux de diplômes les plus faibles auraient tendance à s’adapter au prix de l’immobilier - peut-être en habitant hors des zones où la plupart des habitants ont au moins un Bac+2.")
       st.write("**-R elation entre les entreprises et les salaires.** Les corrélations entre total_entreprise, Micro_entreprise, Petite_entreprise, Moyenne_entreprise, et Grande_entreprise avec les différentes catégories de salaires sont en général faibles, indiquant qu'il n'y a pas de relation directe significative entre le nombre d'entreprises et les niveaux de salaires des différentes catégories de travailleurs.")
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
       st.write(f"Statistique F : {anova_result.statistic}")
       st.write(f"P-value : {anova_result.pvalue}")
       st.write("Les différences de prix au m² entre les régions sont statistiquement significatives.")
       
       st.write("**Interprétation :**")
       st.write("Statistique F élevée (190.91) : Cela signifie qu'il y a une grande différence entre les moyennes des prix au m² dans les différentes régions par rapport à la variance à l'intérieur de chaque région. Plus la statistique F est grande, plus la différence entre les groupes est importante.")
       st.write("P-value de 0.0 : Une p-value de 0.0 signifie que la probabilité d'observer ces différences par hasard est extrêmement faible. Cela nous permet de rejeter l'hypothèse nulle, qui stipule qu'il n'y a pas de différence entre les prix au m² des différentes régions.")

       st.write("**Conclusion :**")
       st.write("Les différences entre les prix moyens au m² des différentes régions sont statistiquement significatives. Autrement dit, le prix de l'immobilier varie de manière significative d'une région à l'autre. Nous pouvons en conclure que les régions n'ont pas des prix immobiliers homogènes et que certains facteurs régionaux influencent les prix au m².")
        
    if option == 'La régression linéaire':
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
    
   # Clustering avant ACP
   # Méthode du coude 
   st.write("Méthode du coude")
    # Afficher l'image 
   image_path = "/Users/morganemuller/Desktop/DATA/Streamlit/Méthode du coude avant ACP.png"  
   st.image(image_path, caption='Méthode du coude avant ACP', use_column_width=True)
   st.write("D'après la méthode du coude, le nombre optimal de clusters semble être 3.")
   # Méthode du dendrogramme
   st.write("Méthode du dendrogramme")
   # Affichage l'image 
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
   # Evaluation de la qualité du clustering
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