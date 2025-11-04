import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Configuration Streamlit
# -----------------------------
st.set_page_config(page_title="Dashboard Arrêts Cardiaques", layout="wide")
st.title("Dashboard des Arrêts Cardiaques")

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

# -----------------------------
# 8️⃣ Corrélations interactives
# -----------------------------
st.subheader("Corrélation avec TARGET pour variables numériques")
num_cols = ['Nb days degraded Physical health',
                 'Nb days degraded mental health','Nb days degraded health'
                 ,'Sleep Time','Weight','Height',
            'Memory loss','Nb days/month drink','Nb drinks/month','Max drinks']

cols_selected = st.multiselect("Choisir variables pour la corrélation", num_cols, default=num_cols[:5])
if cols_selected:
    corr = df_filtered[cols_selected + ['Heart Attack']].corr()['Heart Attack'].sort_values(ascending=False)
    st.bar_chart(corr)

# -----------------------------
# 9️⃣ Boxplots interactifs
# -----------------------------
st.subheader("Boxplots de variables de santé par TARGET")
box_cols = ['Nb days degraded Physical health',
                 'Nb days degraded mental health','Nb days degraded health'
                 ,'Sleep Time']
box_selected = st.multiselect("Choisir colonnes pour boxplots", box_cols, default=box_cols[:3])

for col in box_selected:
    fig, ax = plt.subplots()
    df_filtered.boxplot(column=col, by='Heart Attack', ax=ax)
    ax.set_title(f"{col} par Heart Attack")
    ax.set_xlabel("Heart Attack")
    st.pyplot(fig)
