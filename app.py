import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# CONFIGURATION STREAMLIT
st.set_page_config(page_title="Hackathon Los Tigros 🐅", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Visualisation : Los Tigros 🐅</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    div.stButton {text-align:center;} /* centre le bouton dans sa colonne */
    div.stButton > button {
        height: 70px;
        width: 220px;
        font-size: 20px;
        font-weight: 600;
        border-radius: 12px;
        background-color: #f0f2f6;
        color: #333333;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# COLONNES A GARDER ET MAPPINGS
columns_to_keep = [
    '_STATE', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'PRIMINSR', 
    'PERSDOC3', 'MEDCOST1', 'CHECKUP1', 'EXERANY2', 'SLEPTIM1', 'CVDSTRK3', 
    'CHCSCNC1', 'CHCOCNC1', 'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'DIABETE4', 
    'VETERAN3', 'WEIGHT2', 'HEIGHT3', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 
    'CERVSCRN', 'CRVCLPAP', 'CIMEMLOS', 'CAREGIV1', 'SDHISOLT', 'SDHEMPLY', 
    'MARIJAN1', 'ASBIDRNK', 'CRVCLHPV', 'HADHYST2', 'COLNSIGM', 'COLNCNCR', 
    'VIRCOLO1', 'SMALSTOL', 'STOOLDN2', 'USENOW3', 'ECIGNOW2', 'LCSCTSC1', 
    'LCSSCNCR', 'ALCDAY4', 'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS', 'HIVRISK5',
    'COVIDPOS', 'COVIDSMP', 'PDIABTS1', 'PREDIAB2', 'DIABTYPE', 'COPDBRTH', 
    'COPDBTST', 'COPDSMOK', 'CNCRDIFF', 'CSRVPAIN', 'PSATEST1', '_RFHLTH',
    '_PHYS14D', '_MENT14D', '_HLTHPLN', '_HCVU652', '_TOTINDA', '_ASTHMS1',
    '_SEX', '_AGEG5YR', '_CURECI2', '_PACKDAY', '_SMOKGRP', '_LCSREC', 
    '_RFDRHV8', '_FLSHOT7', '_PNEUMO3', 'ID','TARGET'
]

names_columns = ['State','General health','Nb days degraded Physical health',
                 'Nb days degraded mental health','Nb days degraded health','Source Health Insurance',
                 'Personal Doctor','Could Not Afford Doctor','Last Checkup','Physical Activity',
                 'Sleep Time','Stroke','Skin Cancer','Mélanome','COPD','Depression','Kidney disease',
                 'Diabete','Veteran','Weight','Height','Walk difficulty','Dressing Difficulty','Difficulty Alone',
                 'Cervical Screening','Pap test','Memory Loss','Caregiver','Socially Isolated','Employment Lost',
                 'Marijuana','Drinking asked','HPV test','Hysterectomy','Colonoscopy','Colorectal test',
                 'Virtual colonoscopy','Stool test 1','Stool test 2','Chewing tobacco','E-cigarettes',
                 'Chest scan','Lung cancer test','Nb days/month drink','Nb drinks/month','Nb cuitasses',
                 'Max drinks','HIV risk','COVID positive','COVID symptoms','Last Diabete test',
                 'Prediabete','Diabete type','Breath shortness','Breathing test','Years smoking',
                 'Nb different cancers','Cancer pain','PSA test','Calculated General Health',
                 'Calculated Physical Health','Calculated Mental Health','Calculated Health Insurance',
                 'Adults Health Insurance','Calculated Physical activity','Calculated Asthma status',
                 'Sex','Age category','Current e-cigarette user','Nb cigarettes packs/day','Smoking group',
                 'Lung cancer recommendation','Heavy drinker','Flu shot','Pneumonia vaccination','ID',
                 'Heart Attack']

state_mapping = {1: "Alabama",2: "Alaska",4: "Arizona",5: "Arkansas",6: "California",8: "Colorado",9: "Connecticut",
10: "Delaware",11: "District of Columbia",12: "Florida",13: "Georgia",15: "Hawaii",16: "Idaho",17: "Illinois",
18: "Indiana",19: "Iowa",20: "Kansas",21: "Kentucky",22: "Louisiana",23: "Maine",24: "Maryland",25: "Massachusetts",
26: "Michigan",27: "Minnesota",28: "Mississippi",29: "Missouri",30: "Montana",31: "Nebraska",32: "Nevada",
33: "New Hampshire",34: "New Jersey",35: "New Mexico",36: "New York",37: "North Carolina",38: "North Dakota",
39: "Ohio",40: "Oklahoma",41: "Oregon",42: "Pennsylvania",44: "Rhode Island",45: "South Carolina",
46: "South Dakota",47: "Tennessee",48: "Texas",49: "Utah",50: "Vermont",51: "Virginia",53: "Washington",
54: "West Virginia",55: "Wisconsin",56: "Wyoming",66: "Guam",72: "Puerto Rico",78: "Virgin Islands"}

state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
    'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN',
    'Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
    'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO',
    'Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
    'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
    'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

sex_mapping = {1: "Male",2: "Female"}

# CHARGEMENT DES DONNEES
@st.cache_data
def load_data(n_samples=20000):
    # Charger le CSV
    df = pd.read_csv("data/train.csv", usecols=columns_to_keep)
    
    # Appliquer un échantillonnage stratifié si nécessaire
    if len(df) > n_samples:
        # Colonnes pour la stratification
        strata_cols = ['_SEX', '_AGEG5YR', 'TARGET']
        
        # Supprimer les lignes avec valeurs manquantes sur ces colonnes
        df = df.dropna(subset=strata_cols)
        
        # Fraction à échantillonner
        frac = n_samples / len(df)
        
        # Échantillonnage stratifié
        df, _ = train_test_split(
            df,
            test_size=1-frac,
            stratify=df[strata_cols],
            random_state=42
        )

    return df


df = load_data()
df.columns = names_columns
df['State'] = df['State'].apply(lambda x: state_mapping.get(int(x)) if pd.notna(x) else None)
df['Sex'] = df['Sex'].apply(lambda x: sex_mapping.get(int(x)) if pd.notna(x) else None)
st.write(f"Dataset chargé : {len(df)} lignes")

# CHARTE GRAPHIQUE
cmap = cm.get_cmap("YlOrRd")  # colormap
# On prend les couleurs de 0.3 à 1.0 pour éviter les jaunes très clairs
palette_gradient = [cmap(0.3 + 0.7*i/13) for i in range(14)]
cmap_for_seaborn = mcolors.LinearSegmentedColormap.from_list("custom_gradient", palette_gradient)
palette_hex = [mcolors.to_hex(c) for c in palette_gradient]

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# FILTRES INTERACTIFS
col1,col2,col3 = st.columns([4,2,4])

with col1 :
    states = st.multiselect("Choisir les états", df['State'].unique(), default=df['State'].unique())
    df_filtered = df[df['State'].isin(states)]

with col2 :
    sex_filter = st.multiselect("Choisir le sexe", df['Sex'].unique(), default=df['Sex'].unique())
    df_filtered = df_filtered[df_filtered['Sex'].isin(sex_filter)]

with col3 :
    age_groups_mapping = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
        5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
        9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
        13: "80+", 14: "Missing"
    }
    df_filtered['age_group'] = df_filtered['Age category'].map(age_groups_mapping)
    age_selected = st.multiselect("Choisir tranche(s) d'âge", list(age_groups_mapping.values())[:-1],
                                default=list(age_groups_mapping.values())[:-1])
    df_filtered = df_filtered[df_filtered['age_group'].isin(age_selected)]

# BOUTONS DE NAVIGATION
col1, col2, col3 = st.columns([1,1,1])

with col1:
    bouton_general = st.button("Général")
    if bouton_general :
        st.session_state.page = "general"
    bouton_habitudes = st.button("Habitudes de vie")
    if bouton_habitudes :
        st.session_state.page = "habitudes"
    bouton_social = st.button("Social")
    if bouton_social :
        st.session_state.page = "social"

with col3:
    bouton_sante = st.button("Problèmes de santé")
    if bouton_sante :
        st.session_state.page = "sante"
    bouton_predictions = st.button("Prédictions")
    if bouton_predictions :
        st.session_state.page = "predictions"
    bouton_surprise = st.button("Surprise")
    if bouton_surprise :
        st.session_state.page = "surprise"


# Valeur par défaut au premier chargement
if "page" not in st.session_state:
    st.session_state.page = "general"

if st.session_state.page == "general" :

    col1, col2 = st.columns([1,3])

    with col1 :
        
        # Distribution globale des arrêts cardiaques
        st.subheader("Distribution globale des arrêts cardiaques")
        target_count = df_filtered['Heart Attack'].value_counts(normalize=True).reset_index()
        target_count.columns = ['Heart Attack','Count']

        palette_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
               for r, g, b, a in palette_gradient]

        chart = alt.Chart(target_count).mark_bar().encode(
            x='Heart Attack:N',
            y='Count:Q',
            tooltip=['Heart Attack','Count'],
            color=alt.Color(
                            'Heart Attack:N',
                            scale=alt.Scale(
                                domain=[False,True],
                                range=[palette_hex[0], palette_hex[-1]]
                            ),
        )).properties(width=100, height=400)
        st.altair_chart(chart, use_container_width=True)

    with col2 :

        # Carte par état
        st.subheader("Taux d'arrêts cardiaques par état")

        state_rate = df_filtered.groupby('State')['Heart Attack'].mean().reset_index()
        state_rate.rename(columns={'Heart Attack':'Taux_CVD'}, inplace=True)
        state_rate['State_code'] = state_rate['State'].map(state_abbrev)

        fig = px.choropleth(
            state_rate,
            locations='State_code',       
            locationmode='USA-states',
            color='Taux_CVD',
            color_continuous_scale='Reds',
            scope="usa"
        )
        st.plotly_chart(fig)

    col1, col2 = st.columns([3,1])

    with col1 :

        # Pyramide des âges par sexe et crises cardiaques
        age_mapping = {
            1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
            5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
            9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
            13: "80+", 14: "Missing"
        }
        df_filtered['age_group'] = df_filtered['Age category'].map(age_mapping)
        df_filtered = df_filtered[df_filtered['age_group'] != 'Missing']

        all_combinations = pd.MultiIndex.from_product(
            [list(age_mapping.values())[:-1], ['Male','Female'], [0,1]],
            names=['age_group','Sex','Heart Attack']
        )

        df_counts = df_filtered.groupby(['age_group','Sex','Heart Attack']).size().reindex(all_combinations, fill_value=0).reset_index(name='Count')

        def compute_x(row, df_counts):
            if row['Sex'] == 'Male':
                if row['Heart Attack'] == 1:
                    return -row['Count']  # Crise centrée
                else:
                    # Non-crise empilée à gauche
                    male_crise = df_counts[(df_counts['age_group']==row['age_group']) & (df_counts['Sex']=='Male') & (df_counts['Heart Attack']==1)]['Count'].values[0]
                    return -row['Count'] - male_crise
            else:  # Female
                if row['Heart Attack'] == 1:
                    return row['Count']   # Crise centrée
                else:
                    female_crise = df_counts[(df_counts['age_group']==row['age_group']) & (df_counts['Sex']=='Female') & (df_counts['Heart Attack']==1)]['Count'].values[0]
                    return row['Count'] + female_crise

        df_counts['x'] = df_counts.apply(lambda row: compute_x(row, df_counts), axis=1)

        color_mapping = {
            'Male_0':'#ffd8a8','Male_1':'#d97706',
            'Female_0':'#d8b7f3','Female_1':'#b91c1c'
        }
        df_counts['color'] = df_counts['Sex'] + '_' + df_counts['Heart Attack'].astype(str)
        df_counts['color'] = df_counts['color'].map(color_mapping)

        pyramide_chart = alt.Chart(df_counts).mark_bar().encode(
            y=alt.Y('age_group:N', sort=list(age_mapping.values())[:-1], title='Tranche d’âge'),
            x=alt.X('x:Q', title='Nombre d’individus'),
            color=alt.Color('color:N', scale=None),
            tooltip=['age_group','Sex','Heart Attack','Count']
        ).properties(width=500, height=600)

        st.subheader("Pyramide des âges par sexe et crises cardiaques")
        st.altair_chart(pyramide_chart, use_container_width=True)

    with col2 :

        # Répartition des individus par tranche d'âge
        df_age_counts = df_filtered.copy()
        df_age_counts['age_group'] = df_age_counts['Age category'].map(age_mapping)
        age_counts = df_age_counts['age_group'].value_counts().reindex(age_mapping.values()).reset_index()
        age_counts.columns = ['age_group','Count']

        palette_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r,g,b,_ in palette_gradient]

        age_bar = alt.Chart(age_counts).mark_bar().encode(
            y=alt.Y('age_group:N', sort=list(age_mapping.values())[:-1], title='Tranche d’âge'),
            x=alt.X('Count:Q', title='Nombre d’individus'),
            color=alt.Color('Count:Q', scale=alt.Scale(range=palette_hex)),
            tooltip=['age_group','Count']
        ).properties(width=400, height=500)

        st.subheader("Répartition des individus par tranche d'âge")
        st.altair_chart(age_bar)


    nan_proportion = df_filtered.isna().sum().sort_values(ascending=False) / len(df_filtered)
    norm = mcolors.Normalize(vmin=nan_proportion.min(), vmax=nan_proportion.max())

    # Générer la couleur de chaque barre selon la valeur
    colors = [cmap(norm(v)) for v in nan_proportion.values]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(nan_proportion.index, nan_proportion.values, color=colors)

    plt.xticks(rotation=90,fontsize=4)
    plt.ylabel("Proportion de NaN")
    plt.title("Proportion de valeurs manquantes par colonne")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Ajouter une colorbar pour la lisibilité
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    st.pyplot(fig)


if st.session_state.page == "habitudes" :

    habitude_vars = [
        'Physical Activity', 'Sleep Time', 'Nb days/month drink', 'Nb drinks/month',
        'Marijuana','Drinking asked','Smoking group','Flu shot','Pneumonia vaccination'
    ]

    st.subheader("Analyse des habitudes de vie")
    selected_vars = st.multiselect(
        "Choisir les variables d'habitudes de vie à analyser",
        habitude_vars,
        default=habitude_vars[:3]
    )

    if selected_vars:

        col1, col2 = st.columns([2,2])

        with col1:
            st.subheader("Distributions et taux par catégorie")
            for var in selected_vars:
                st.markdown(f"**{var}**")

                cola, colb = st.columns([1,1])

                with cola :
                    counts = df_filtered[var].value_counts().reset_index()
                    counts.columns = [var, 'count']
                    counts['color'] = [palette_hex[i % len(palette_hex)] for i in range(len(counts))]
                    bar = alt.Chart(counts).mark_bar().encode(
                        x=alt.X(f'{var}:N'),
                        y=alt.Y('count:Q'),
                        color=alt.Color('color:N', scale=None),
                        tooltip=[var,'count']
                    ).properties(width=350, height=150)
                    st.altair_chart(bar, use_container_width=True)

                with colb :

                    # Taux de Heart Attack par catégorie
                    if var in df_filtered.columns:
                        rate = df_filtered.groupby(var)['Heart Attack'].mean().reset_index()
                        rate['color'] = [palette_hex[i % len(palette_hex)] for i in range(len(rate))]
                        rate_chart = alt.Chart(rate).mark_bar(color='#d9534f').encode(
                            x=alt.X(f'{var}:N'),
                            y=alt.Y('Heart Attack:Q', title='Taux de Heart Attack'),
                            color=alt.Color('color:N', scale=None),
                            tooltip=[var,'Heart Attack']
                        ).properties(width=350, height=150)
                        st.altair_chart(rate_chart, use_container_width=True)

        with col2:
            st.subheader("Matrice de corrélation avec la target")
            numeric_vars = [v for v in selected_vars if pd.api.types.is_numeric_dtype(df_filtered[v])]
            corr_vars = numeric_vars + ['Heart Attack']

            if len(corr_vars) > 1:
                corr_matrix = df_filtered[corr_vars].corr()
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
                st.pyplot(fig)
            else:
                st.write("Pas assez de variables numériques sélectionnées pour la corrélation.")


if st.session_state.page == "social" :

    social_vars = [
        'Source Health Insurance',
        'Personal Doctor',
        'Could Not Afford Doctor',
        'Last Checkup',
        'Adults Health Insurance',
        'Calculated Health Insurance',
        'Caregiver',
        'Socially Isolated',
        'Employment Lost'
    ]

    st.subheader("Analyse des critères sociaux")

    # Multiselect
    selected_social = st.multiselect(
        "Choisir les variables sociales à analyser",
        social_vars,
        default=social_vars[:3]
    )

    if selected_social:

        col1, col2 = st.columns([2,2])

        # -----------------------------
        # Colonne gauche : distributions et taux
        # -----------------------------
        with col1:
            st.subheader("Distributions et taux de Heart Attack par catégorie")
            
            for var in selected_social:
                st.markdown(f"**{var}**")
                cola, colb = st.columns([1,1])

                with cola:
                    # Barplot pour variables catégorielles
                    counts = df_filtered[var].value_counts().reset_index()
                    counts.columns = [var, 'count']
                    bar = alt.Chart(counts).mark_bar().encode(
                        x=alt.X(f'{var}:N'),
                        y=alt.Y('count:Q'),
                        tooltip=[var,'count']
                    ).properties(width=350, height=150)
                    st.altair_chart(bar, use_container_width=True)

                with colb:
                    # Taux de Heart Attack par catégorie
                    rate = df_filtered.groupby(var)['Heart Attack'].mean().reset_index()
                    rate_chart = alt.Chart(rate).mark_bar(color='#d9534f').encode(
                        x=alt.X(f'{var}:N'),
                        y=alt.Y('Heart Attack:Q', title='Taux de Heart Attack'),
                        tooltip=[var,'Heart Attack']
                    ).properties(width=350, height=150)
                    st.altair_chart(rate_chart, use_container_width=True)

        # -----------------------------
        # Colonne droite : corrélation
        # -----------------------------
        with col2:
            st.subheader("Matrice de corrélation avec la target")
            numeric_social_vars = [v for v in selected_social if pd.api.types.is_numeric_dtype(df_filtered[v])]
            corr_vars = numeric_social_vars + ['Heart Attack']

            if len(corr_vars) > 1:
                corr_matrix = df_filtered[corr_vars].corr()
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
                st.pyplot(fig)
            else:
                st.write("Pas assez de variables numériques pour calculer une corrélation.")


if st.session_state.page == "predictions" :

    top_vars = ['General health','Age category','Heavy drinker','Smoking group','Sleep Time']

    # Préparer les données
    df_pca = df_filtered[top_vars + ['Heart Attack']].dropna()
    if len(df_pca) > 2000:
        df_pca = df_pca.sample(2000, random_state=42)

    X_scaled = StandardScaler().fit_transform(df_pca[top_vars])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot['Heart Attack'] = df_pca['Heart Attack'].values

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: '#1f77b4', 1: '#d9534f'}
    for ha in df_plot['Heart Attack'].unique():
        subset = df_plot[df_plot['Heart Attack'] == ha]
        ax.scatter(subset['PC1'], subset['PC2'],
                   label=f'Heart Attack={ha}', alpha=0.6, c=colors[ha])

    # Flèches des variables
    for i, var in enumerate(top_vars):
        ax.arrow(0, 0,
                 pca.components_[0, i] * 3,
                 pca.components_[1, i] * 3,
                 color='black', alpha=0.7, head_width=0.1)
        ax.text(pca.components_[0, i] * 3.2,
                pca.components_[1, i] * 3.2,
                var, fontsize=9)

    ax.axhline(0, color='grey')
    ax.axvline(0, color='grey')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ACP sur les variables les plus informatives (Mutual Information)")
    ax.legend()
    st.pyplot(fig)


if st.session_state.page == "surprise" :

    st.balloons()
