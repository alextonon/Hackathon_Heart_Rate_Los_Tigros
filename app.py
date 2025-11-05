import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.font_manager as fm

# -----------------------------
# 1️⃣ Configuration Streamlit
# -----------------------------
st.set_page_config(page_title="Hackathon Los Tigros 🐅", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Visualisation : Los Tigros 🐅</h1>", unsafe_allow_html=True)
# -----------------------------
# 2️⃣ Colonnes à garder
# -----------------------------
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

# -----------------------------
# 3️⃣ Charger les données
# -----------------------------
@st.cache_data
def load_data(file_path, n_samples=10000):
    df = pd.read_csv(file_path, usecols=columns_to_keep)
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42)
    return df

file_path = "data/train.csv"
df = load_data(file_path)
df.columns = names_columns
st.write(f"Dataset chargé : {len(df)} lignes")
st.dataframe(df.head(5))

# -----------------------------
# 1️⃣ Créer un dégradé rouge-orangé-jaune foncé (éviter les jaunes clairs)
# -----------------------------
cmap = cm.get_cmap("YlOrRd")  # colormap
# On prend les couleurs de 0.3 à 1.0 pour éviter les jaunes très clairs
palette_gradient = [cmap(0.3 + 0.7*i/13) for i in range(14)]

# -----------------------------
# 2️⃣ Police et style global
# -----------------------------
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# -----------------------------
# 4️⃣ Filtres interactifs
# -----------------------------
states = st.multiselect("Choisir les états", df['State'].unique(), default=df['State'].unique())
df_filtered = df[df['State'].isin(states)]

sex_filter = st.multiselect("Choisir le sexe", [1,2], default=[1,2])
df_filtered = df_filtered[df_filtered['Sex'].isin(sex_filter)]

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

# -----------------------------
#  Boutons de navigation persistants
# -----------------------------
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    if st.button("Général"):
        st.session_state.page = "social"

with col2:
    if st.button("Habitudes"):
        st.session_state.page = "habitudes"

with col3:
    if st.button("Corrélations"):
        st.session_state.page = "correlations"

with col4:
    if st.button("Surprise"):
        st.session_state.page = "surprise"

# Valeur par défaut au premier chargement
if "page" not in st.session_state:
    st.session_state.page = "social"

if st.session_state.page == "social" :

    # -----------------------------
    # 5️⃣ Distribution globale de TARGET
    # -----------------------------
    st.subheader("Distribution globale des arrêts cardiaques")
    target_count = df_filtered['Heart Attack'].value_counts().reset_index()
    target_count.columns = ['Heart Attack','Count']

    chart = alt.Chart(target_count).mark_bar().encode(
        x='Heart Attack:N',
        y='Count:Q',
        tooltip=['Heart Attack','Count'],
        color='Heart Attack:N'
    ).properties(width=400, height=300)
    st.altair_chart(chart, use_container_width=True)

    # -----------------------------
    # 6️⃣ Pyramide par âge et sexe (simplifiée)
    # -----------------------------
    st.subheader("Pyramide des âges par sexe")
    counts_male = []
    counts_female = []
    age_order = list(age_groups_mapping.values())[:-1]

    for age in age_order:
        group = df_filtered[df_filtered['age_group']==age]
        male = group[group['Sex']==1]['Heart Attack'].value_counts()
        female = group[group['Sex']==2]['Heart Attack'].value_counts()
        counts_male.append((male.get(0,0), male.get(1,0)))
        counts_female.append((female.get(0,0), female.get(1,0)))

    fig, ax = plt.subplots(figsize=(10,7))
    for i, age in enumerate(age_order):
        male_no, male_yes = counts_male[i]
        ax.barh(age, -male_yes, color="#d97706", edgecolor='white')
        ax.barh(age, -male_no, left=-male_yes, color="#ffedd5", edgecolor='white')
        female_no, female_yes = counts_female[i]
        ax.barh(age, female_yes, color="#b91c1c", edgecolor='white')
        ax.barh(age, female_no, left=female_yes, color="#fcd5d5", edgecolor='white')

    ax.axvline(x=0, color='white', linewidth=2)
    ax.set_xlim(-max(sum(c) for c in counts_male)*1.1, max(sum(c) for c in counts_female)*1.1)
    ax.set_xlabel("Nombre d'individus")
    ax.set_ylabel("Tranche d'âge")
    st.pyplot(fig)

    # -----------------------------
    # 3️⃣ Mapping des codes vers tranches d'âge
    # -----------------------------
    age_mapping = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
        5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
        9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
        13: "80+", 14: "Missing"
    }

    # -----------------------------
    # 4️⃣ Préparer le DataFrame
    # -----------------------------
    df_age = df_filtered[['Age category']].copy()
    df_age['age_group'] = df_age['Age category'].map(age_mapping)

    # Compter les individus par tranche et garder l'ordre
    age_counts = df_age['age_group'].value_counts().reindex(age_mapping.values())

    # -----------------------------
    # 5️⃣ Graphe Seaborn avec dégradé
    # -----------------------------
    fig = plt.figure(figsize=(12,6))
    sns.barplot(x=age_counts.index, y=age_counts.values, palette=palette_gradient)

    # Titres et labels
    plt.xlabel("Tranche d'âge", fontsize=14, fontweight='bold', style='italic')
    plt.ylabel("Nombre d'individus", fontsize=14, fontweight='bold')
    plt.title("Répartition des individus par tranche d'âge", fontsize=18, fontweight='bold')

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)


    # -----------------------------
    # 7️⃣ Taux par état
    # -----------------------------
    st.subheader("Taux d'arrêts cardiaques par état")
    state_rate = df_filtered.groupby('State')['Heart Attack'].mean().reset_index()
    state_rate.rename(columns={'Heart Attack':'Taux_CVD'}, inplace=True)
    chart = alt.Chart(state_rate).mark_bar().encode(
        x=alt.X('State:N', sort='-y'),
        y='Taux_CVD:Q',
        tooltip=['State','Taux_CVD']
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)


if st.session_state.page == "correlations" :

    # -----------------------------
    # 8️⃣ Corrélations interactives
    # -----------------------------
    st.subheader("Corrélation avec TARGET pour variables numériques")

    # -----------------------------
    # 1️⃣ Variables continues à analyser
    # -----------------------------
    selected_vars = st.multiselect("Choisir variables à considérer",options=['Nb days degraded Physical health',
                 'Nb days degraded mental health','Sleep Time', 'Weight', 'Height'])

    # -----------------------------
    # 2️⃣ Préparer le DataFrame
    # -----------------------------
    df_corr = df[selected_vars + ['Heart Attack']].copy()

    # Supprimer les lignes avec valeurs manquantes
    df_corr = df_corr.dropna()

    # -----------------------------
    # 3️⃣ Calculer la corrélation
    # -----------------------------
    corr_matrix = df_corr.corr()

    # -----------------------------
    # 4️⃣ Heatmap
    # -----------------------------
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, linecolor='white')
    plt.title("Corrélation entre variables continues et crise cardiaque", fontsize=16, fontweight='bold')
    st.pyplot(fig)

    # # -----------------------------
    # # 9️⃣ Boxplots interactifs
    # # -----------------------------
    # st.subheader("Boxplots de variables de santé par TARGET")
    # box_cols = ['Nb days degraded Physical health',
    #                 'Nb days degraded mental health','Nb days degraded health'
    #                 ,'Sleep Time']
    # box_selected = st.multiselect("Choisir colonnes pour boxplots", box_cols, default=box_cols[:3])

    # for col in box_selected:
    #     fig, ax = plt.subplots()
    #     df_filtered.boxplot(column=col, by='Heart Attack', ax=ax)
    #     ax.set_title(f"{col} par Heart Attack")
    #     ax.set_xlabel("Heart Attack")
    #     st.pyplot(fig)


if st.session_state.page == "surprise" :

    st.balloons()
