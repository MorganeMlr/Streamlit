# # Diagnostic information
# st.write(f"Python executable: {sys.executable}")
# st.write(f"Python version: {sys.version}")
# st.write("Installed packages:")
# st.code(subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8"))

import sys
sys.path.append('src')

import streamlit as st
import pandas as pd


st.title("French Industry Project")

###SIDEBAR ###
# Add the project logo 
logo_url = "https://drive.google.com/file/d/1PbL3o3ZW_YK3fOARIIjW6SsWtmkuJJF4"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Contexte et objectifs du projet", "Le Jeu De Données", "Data Vizualization", "Préparation des données" ,"Modélisation", "Analyse des Résultats", "Conclusions & Perspectives"]
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
              **Dans quelle mesure la géographie influence-t-elle les inégalités de salaires en France ?**
              **La géographie**, dans le cadre de cette problématique, concerne l'étude des facteurs spatiaux et régionaux qui influencent les conditions économiques et sociales des populations et des entreprises. Elle examine comment les caractéristiques physiques, les infrastructures, et les dynamiques locales, notamment économiques, impactent le développement et les niveaux de salaire.
              **Les inégalités de salaires**, elles, se réfèrent aux différences de rémunération perçues par les travailleurs en fonction de divers facteurs tels que l'expérience, l'éducation, le secteur d'activité, le genre, et, dans ce cadre, la localisation géographique. Ces disparités peuvent se manifester à travers des écarts de salaire brut, des primes, et des avantages sociaux.
              
              Ce projet est réalisé dans le cadre de la formation Data Analyst de DataScientest et ne s’insère pas directement dans les métiers respectifs des membres du groupe. Il s’agit pour nous d’une première, et nous sommes enthousiastes à l’idée de découvrir de près le métier de data analyst. 
              En revanche, ce projet pourrait facilement s’insérer dans le métier d’un data analyst travaillant pour une institution publique (par exemple l’Observatoire des inégalités, une collectivité territoriale voulant en connaître plus sur les inégalités de salaire dans sa localité, ou encore un ministère - à titre d’exemple le ministère de la Cohésion des territoires ou le ministère du Travail)  souhaitant proposer des politiques publiques pour réduire ces inégalités.

              **Du point de vue technique**, les données sont en effet collectées et traitées dans un premier temps par l’INSEE, avant d’être analysées statistiquement et modélisées / visualisées par des agents de cet institut ou d’un autre organisme public. 
              **Du point de vue économique**, les organismes publics mentionnées ci-dessus ont besoin de réaliser des analyses des impacts économiques de ces inégalités afin de proposer de nouvelles politiques publiques visant à réduire ces dernières avant enfin d’évaluer le résultat de ces mêmes politiques publiques. 
              **Du point de vue scientifique**, il serait intéressant de réaliser une revue de littérature car de très nombreuses études et recherches existent déjà sur le sujet. Cela nous permettrait de nous appuyer sur des travaux existants pour affiner nos recherches et nos hypothèses. A l’issue de nos travaux, il serait envisageable de diffuser nos résultats à la communauté scientifique et au grand public, via un rapport ou un article.
              """)
              
    st.markdown("""
            #### Les objectifs de notre analyse
            Les principaux objectifs que nous nous sommes fixés sont les suivants : 
            **- Identifier les disparités salariales horaires régionales : cartographier les inégalités de salaires horaires à travers les différentes régions de France en utilisant des données géographiques et salariales détaillées.**
            **- Analyser les facteurs déterminants des inégalités : déterminer les principaux facteurs géographiques, économiques, sociaux et individuels qui influencent les variations des salaires horaires, tels que l'accès aux infrastructures, le genre, l’âge, etc.**
            **- Faire des propositions de politiques publiques pouvant participer à la réduction de ces inégalités au vu des conclusions de notre étude.**

            Les différents membres du groupe ne sont pas experts des problématiques liées aux inégalités de salaire en France, mais sont curieux d’en apprendre plus sur ce sujet qui concerne directement tous les citoyens. 
            Des projets similaires sont déjà menés par différents organismes publics et les principaux facteurs des inégalités de salaires en France sont déjà connus. Nous ne prétendons pas découvrir de nouveaux facteurs d’inégalités, mais bien comprendre de l’intérieur comment des data analysts ont pu trouver les réponses à ces questions cruciales pour les politiques publiques en France, et pourquoi pas réfléchir à des idées de politiques publiques de réduction des inégalités qui pourraient par la suite s’insérer dans les métiers respectifs des membres de l’équipe. 
            """)

## Le Jeu De Données Page 
if page == pages[1]:
    st.write("### Le jeu de données")
    st.write("Les sources employées pour ce projet proviennent de l’INSEE, Institut national de la statistique et des études économiques ainsi que Data.gouv, qui est une plateforme officielle de l'État français dédiée à la diffusion et au partage des données publiques. Il s'agit donc de données officielles concernant les différents aspects économiques, démographiques et géographiques du territoire français.")

    st.markdown("""
    ### Compréhension et manipulation des données

    Afin d’atteindre les objectifs de notre projet, nous avons utilisé plusieurs jeux de données. 
    De façon générale, ces derniers sont issus de l’INSEE ou de site gouvernementaux. L’INSEE (Institut National de la Statistique et des Etudes Economiques) est l'institut officiel français qui collecte des données de tous types sur le territoire français. Les données que ce dernier collecte et analyse sont disponibles librement, afin de permettre à tous les organismes publics, citoyens et entreprises privées d’accéder à des informations et analyses intéressantes qui peuvent les concerner. 

    Les jeux de données utilisés sont les suivants : 

    - La table **'base_etablissement_par_tranche_effectif.csv'** : Informations sur le nombre d'entreprises dans chaque ville française classées par taille ;
    - La table **'name_geographic_information'**: Données géographiques sur les villes françaises (principalement la latitude et la longitude, mais aussi les codes et les noms des régions/départements);
    - La table **'net_salary_per_town_categories'** : Salaires par villes française par catégories d'emploi, âge et sexe. Ce fichier fournit des données sur les salaires nets horaires. Les indicateurs sont ventilés selon le sexe, l'âge, la catégorie socioprofessionnelle (hors agriculture); 
    - La table **'population.csv'** : Informations démographiques par ville, âge, sexe et mode de vie; 
    - La table **'referentiel-gares-voyageurs.csv'** : Informations sur les gares TGV présentes sur le territoire français. 
    - La table **'aeroports_france'** : Informations sur les aéroports français. 
    """)

    st.write("#### Afficher les données")
    st.dataframe(df)  # Affiche le DataFrame avec la mise en page par défaut

    