# Bibliothèques pour la gestion des données
import pandas as pd
import numpy as np

# Bibliothèques pour les analyses statistiques
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Bibliothèques pour la visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Bibliothèque pour l'interface utilisateur et le tableau de bord
import streamlit as st

# --------- CONFIG ---------
st.set_page_config(page_title="Analyse de Survie", layout="wide")

uploaded_file = st.sidebar.file_uploader("📂 Charger un fichier CSV", type=["csv"])
encoding = st.sidebar.selectbox("🔤 Encodage du fichier", ["utf-8", "latin1"])

@st.cache_data
def load_data(file, encoding):
    return pd.read_csv(file, encoding=encoding)

if uploaded_file is not None:
    df = load_data(uploaded_file, encoding)
else:
    st.warning("Veuillez charger un fichier CSV pour continuer.")
    st.stop()


# --------- CHARGER LE FICHIER ---------
# encoding = st.sidebar.selectbox("Encodage", ["utf-8", "latin1"], key="encoding_selector")


# @st.cache_data
# def load_data():
    # df = pd.read_csv("survival_data_1000.csv", encoding=encoding)
    # return df

# --------- SIDEBAR ---------
st.sidebar.header("🎯 Filtres d'analyse")
variables = ["Age", "Sex", "Smoker", "Comorbidities", "Treatment",
             "BMI", "Physical_Activity", "Time_to_Event", "Event_Observed"]
selected_variables = st.sidebar.multiselect("Variables à inclure :", variables)


# --------- CREATION DES TRANCHES D'AGE ET DE BMI ---------

def create_age_bmi_bins(df):
    # Créer les tranches d'âge
    bins_age = [0, 50, 60, 100]
    labels_age = ["<50", "50-60", ">60"]
    df["Tranche_Age"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age, right=False)

    # Créer les tranches de BMI
    bins_bmi = [0, 18, 26, 50]
    labels_bmi = ["<18", "18-26", ">26"]
    df["Tranche_BMI"] = pd.cut(df["BMI"], bins=bins_bmi, labels=labels_bmi, right=False)

    return df


df = load_data(uploaded_file, encoding)
df = create_age_bmi_bins(df)

# --------- AFFICHAGE DES TRANCHES D'AGE ET DE BMI ---------
st.sidebar.subheader("📌 Sous-groupes par tranche d'âge et de BMI")
age_group = st.sidebar.multiselect("Tranches d'âge",
                                   ["Choose an option"] + list(df["Tranche_Age"].unique())
                                   # Option par défaut "Choose an option"
                                   )
bmi_group = st.sidebar.multiselect("Tranches de BMI",
                                   ["Choose an option"] + list(df["Tranche_BMI"].unique())
                                   # Option par défaut "Choose an option"
                                   )

# -------- MENU NAVIGABLE --------
st.title("📈 Dashboard d'Analyse de Survie ")

sections = [
    "Visualisation des données",
    "Traitement des données",
    "Statistiques descriptives",
    "Représentations graphiques des variables",
    "Probabilités et courbes de survie",
    "Prédiction de survie d’un individu",
    "Modèle de régression de Cox",
    "Résumé interactif",
    "À propos du projet"
]

# --------- MENU DEROULANT EN HAUT ---------
selected_section = st.selectbox("🔍 Choisir une analyse :", sections)

# -------- CONTENU DYNAMIQUE --------
if selected_section == "Visualisation des données":
    st.subheader("🔍 Visualisation des données")

    # --------- APPLIQUER LES FILTRES ---------

    # Des tranches d'âge et de BMI
    if "Choose an option" not in age_group and "Choose an option" not in bmi_group:
        filtered_df = df[
            (df["Tranche_Age"].isin(age_group)) &
            (df["Tranche_BMI"].isin(bmi_group))
            ]
    elif "Choose an option" not in age_group:
        filtered_df = df[df["Tranche_Age"].isin(age_group)]  # Filtrer seulement par les tranches d'âge sélectionnées
    elif "Choose an option" not in bmi_group:
        filtered_df = df[df["Tranche_BMI"].isin(bmi_group)]  # Filtrer seulement par les tranches de BMI sélectionnées
    else:
        filtered_df = df  # Pas de filtre appliqué si l'utilisateur n'a pas sélectionné une option

    # Des variables sélectionnées
    if selected_variables:
        selected_variables = selected_variables + ["Tranche_Age", "Tranche_BMI"]
        filtered_df = filtered_df[selected_variables]

        # Affichage des 10 premières lignes des données filtrées
    st.write("Voici un aperçu des 10 premières lignes de vos données filtrées :")
    st.dataframe(filtered_df.head(10))
    st.markdown(f"**Nombre de lignes sélectionnées :** {len(filtered_df)}")

elif selected_section == "Traitement des données":
    st.subheader("🧼 Traitement des données manquantes")
    st.write("(Infos et outils pour nettoyer les données)")

    # --------- GESTION DES DOUBLONS ---------
    if st.button("Lancer la suppression des doublons"):
        # Colonnes qui définissent un patient
        patient_columns = [
            "Age", "Sex", "Smoker", "Comorbidities", "Treatment",
            "BMI", "Physical_Activity", "Time_to_Event", "Event_Observed"
        ]

        # Marquer les doublons uniquement pour les Event_Observed = 0
        is_duplicate = df["Event_Observed"].eq(0) & df.duplicated(subset=patient_columns)

        # Identifier les lignes marquées comme doublons
        duplicates_to_drop = df[is_duplicate]

        # Supprimer les lignes marquées comme doublons
        df = df[~is_duplicate]

        # Afficher les lignes qui vont être supprimées
        st.write("🔍 Lignes identifiées comme doublons à supprimer :")
        st.write(duplicates_to_drop)

        # Afficher combien on va en supprimer
        num_duplicates = is_duplicate.sum()
        st.write(f"Nombre de doublons supprimés (Event_Observed = 0) : {num_duplicates}")

        # Afficher le dataframe nettoyé
        st.write(df)

        # Voir le nombre de doublons parmi les patients non observés
        st.markdown(f"**Nombre de lignes actuelles dans le jeu de données :** {len(df)}")

    if st.button("Lancer la gestion des données manquantes"):
        st.subheader("📋 Gestion des données manquantes")

        # Affichage des colonnes avec des valeurs manquantes
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]

        if missing_info.empty:
            st.success("✅ Aucune donnée manquante détectée.")
        else:
            st.write("🔍 Colonnes avec données manquantes :")
            st.write(missing_info)

            option = st.selectbox("Méthode de traitement :",
                                  ["Supprimer lignes", "Remplacer par moyenne", "Remplacer par médiane"])

            if st.button("Appliquer le traitement"):
                if option == "Supprimer lignes":
                    df.dropna(inplace=True)
                    st.success("Lignes contenant des NaN supprimées.")
                elif option == "Remplacer par moyenne":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col].fillna(df[col].mean(), inplace=True)
                    st.success("Valeurs manquantes remplacées par la moyenne.")
                elif option == "Remplacer par médiane":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col].fillna(df[col].median(), inplace=True)
                    st.success("Valeurs manquantes remplacées par la médiane.")

                st.write("DataFrame après traitement :")
                st.dataframe(df)


elif selected_section == "Statistiques descriptives":
    st.subheader("📊 Statistiques descriptives")
    st.write(df.describe())

    st.sidebar.markdown("---")
    # Histogramme interactif
    var = st.selectbox("Choisir une variable", df.select_dtypes(include='number').columns)
    fig = px.histogram(df, x=var, nbins=20)
    st.plotly_chart(fig)

elif selected_section == "Représentations graphiques des variables":
    st.subheader("📈 Représentations graphiques des variables")
    st.write("(Histogrammes, boxplots, etc.)")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    st.markdown("### 📊 Histogramme d'une variable numérique")
    num_var = st.selectbox("Choisir une variable numérique :", numeric_cols)
    fig = px.histogram(df, x=num_var, nbins=30, title=f"Histogramme de {num_var}")
    st.plotly_chart(fig)

    st.markdown("### 📦 Boxplot par variable catégorielle")
    if categorical_cols and numeric_cols:
        cat_var = st.selectbox("Variable catégorielle :", categorical_cols)
        num_var2 = st.selectbox("Variable numérique :", numeric_cols, key="boxplot")
        fig2 = px.box(df, x=cat_var, y=num_var2, title=f"Boxplot de {num_var2} par {cat_var}")
        st.plotly_chart(fig2)


elif selected_section == "Probabilités et courbes de survie":
    st.subheader("⏳ Probabilités et courbes de survie")
    st.write("(Kaplan-Meier, etc.)")

    # Vérification des colonnes nécessaires
    if "Time_to_Event" in df.columns and "Event_Observed" in df.columns:
        # --------- ESTIMATION DE KAPLAN-MEIER ---------
        st.subheader("1️⃣ Courbe de survie globale (Kaplan-Meier)")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df["Time_to_Event"], event_observed=df["Event_Observed"])

        # --------- TABLEAU DES PROBABILITES ---------
        st.markdown("### 📌 Probabilités de survie à chaque instant t")
        st.dataframe(kmf.survival_function_.reset_index().rename(columns={
            "timeline": "Temps (t)",
            "KM_estimate": "Probabilité de survie"
        }).head(10))

        # --------- COURBE GLOBALE ---------
        st.markdown("### 📈 Courbe de survie globale avec intervalle de confiance")
        window_width = 550
        fig_width = window_width // 40
        fig_height = 6  # Hauteur standard pour la courbe

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title("Courbe de survie - Kaplan-Meier")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Probabilité de survie")
        st.pyplot(fig)
        plt.close(fig)

        # --------- COURBE PAR GROUPE ---------
        st.subheader("2️⃣ Comparaison selon un critère")
    # On conserve uniquement les variables catégorielles présentes dans le DataFrame
    kmf_vars = ["Sex", "Smoker", "Comorbidities", "Treatment", "Physical_Activity", "Tranche_Age", "Tranche_BMI"]

    if kmf_vars:
        selected_group = st.selectbox("Choisir une variable de regroupement :", kmf_vars)

        # Taille du graphique
        window_width = 550
        fig_width = window_width // 40
        fig_height = 6  # Hauteur standard pour la courbe

        fig_group, ax_group = plt.subplots(figsize=(fig_width, fig_height))
        for group in df[selected_group].dropna().unique():
            mask = df[selected_group] == group
            kmf.fit(df["Time_to_Event"][mask], df["Event_Observed"][mask], label=str(group))
            kmf.plot_survival_function(ax=ax_group, ci_show=True)

        ax_group.set_title(f"Courbe de survie selon {selected_group}")
        ax_group.set_xlabel("Temps")
        ax_group.set_ylabel("Probabilité de survie")
        st.pyplot(fig_group)

        if len(df[selected_group].unique()) == 2:  # Si 2 groupes à comparer
            group1 = df[selected_group] == df[selected_group].unique()[0]
            group2 = df[selected_group] == df[selected_group].unique()[1]
            results = logrank_test(
                df["Time_to_Event"][group1], df["Time_to_Event"][group2],
                df["Event_Observed"][group1], df["Event_Observed"][group2]
            )
            st.write(f"**Test de log-rank** : p-value = {results.p_value:.4f}")
        st.markdown("---")

    else:
        st.warning("Aucune variable catégorielle de regroupement disponible.")


elif selected_section == "Prédiction de survie d’un individu":
    st.subheader("🤖 Prédiction de survie d’un individu")
    st.write("(Modèle prédictif appliqué à un individu)")

    # Vérification des colonnes nécessaires
    if "Time_to_Event" in df.columns and "Event_Observed" in df.columns:
        naf = NelsonAalenFitter()
        naf.fit(durations=df["Time_to_Event"], event_observed=df["Event_Observed"])

        # --------- COURBE DU RISQUE CUMULÉ ---------
        st.markdown("### 📈 Courbe du risque cumulé")

        # Taille du graphique
        window_width = 550
        fig_width = window_width // 40
        fig_height = 6  # Hauteur standard pour la courbe

        fig_risk, ax_risk = plt.subplots(figsize=(fig_width, fig_height))
        naf.plot_cumulative_hazard(ax=ax_risk)
        ax_risk.set_title("Courbe du risque cumulé (Nelson-Aalen)")
        ax_risk.set_xlabel("Temps")
        ax_risk.set_ylabel("Risque cumulé")
        st.pyplot(fig_risk)

        # --------- EXPLICATION DE L’ÉVOLUTION DU RISQUE ---------
        st.markdown("""
        ### 📌 Explication de l’évolution du risque en fonction du temps
        La courbe du risque cumulée montre l'évolution du risque qu'un événement se produise au fil du temps. 
        Au fur et à mesure que le temps passe, le risque cumulatif peut augmenter, indiquant que le risque 
        d'un événement devient plus probable au fur et à mesure que l'on avance dans le temps.
        """)  # A refaire

        # --------- ESTIMATION DE LA SURVIE POUR UN TEMPS DONNÉ ---------
        st.subheader("2️⃣ Estimation de la survie pour un temps donné")
        time_input = st.number_input("Entrez un temps pour estimer la survie :", min_value=0, value=12)

        kmf = KaplanMeierFitter()
        kmf.fit(durations=df["Time_to_Event"], event_observed=df["Event_Observed"])

        # Calcul de la probabilité de survie
        survival_probability = kmf.predict(time_input)

        # Affichage de l'estimation de survie à ce temps
        st.markdown(f"### 📊 Estimation de la survie à t={time_input} : {survival_probability:.3f}")

        # --------- EXPLICATION DE L’ÉVOLUTION DU RISQUE ---------
        st.markdown("""
            ### 📌 Explication de la méthode de Nelson-Aalen et de la courbe de risque cumulé

            La méthode de Nelson-Aalen est utilisée pour estimer la fonction de risque cumulé à partir de données de survie. 
            Contrairement à la méthode de Kaplan-Meier qui estime la probabilité de survie, Nelson-Aalen se concentre sur le risque d'événement au fil du temps.

            **Fonction de Risque Cumulé :**

            La fonction de risque cumulé, estimée par la méthode de Nelson-Aalen, représente le risque total qu'un événement (comme le décès ou la défaillance) se soit produit jusqu'à un certain moment. 
            Elle est calculée en additionnant les risques instantanés à chaque instant où un événement se produit.

            **Interprétation de la Courbe :**

            -   **Courbe Ascendante :** Une courbe ascendante indique que le risque cumulé augmente avec le temps. 
                Cela signifie que la probabilité qu'un événement se produise s'accroît à mesure que le temps avance.
            -   **Pente Raide :** Une pente raide suggère une augmentation rapide du risque, indiquant que les événements se produisent plus fréquemment dans cet intervalle de temps.
            -   **Pente Douce :** Une pente douce indique une augmentation plus lente du risque, suggérant que les événements sont moins fréquents.
            -   **Plateaux :** Les plateaux peuvent indiquer des périodes où le risque reste relativement constant.

            **Implications :**

            -   **Maladies :** Dans le contexte médical, une augmentation du risque cumulé peut signifier que la condition du patient se détériore avec le temps, augmentant la probabilité de complications ou de décès.
            -   **Fiabilité :** En ingénierie, cela peut indiquer une augmentation de la probabilité de défaillance d'un composant avec l'usure.

            **En résumé,** la méthode de Nelson-Aalen et la courbe de risque cumulé fournissent des informations précieuses sur la dynamique du risque dans les analyses de survie, 
            complétant ainsi les informations obtenues par les courbes de survie de Kaplan-Meier.
            """)

        # --------- AFFICHAGE DE LA COURBE DE SURVIE ---------
        st.subheader("📈 Courbe de survie Kaplan-Meier")

        # Taille du graphique
        window_width = 550
        fig_width = window_width // 40
        fig_height = 6  # Hauteur standard pour la courbe

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        kmf.plot_survival_function(ax=ax)
        ax.set_title("Courbe de survie - Kaplan-Meier")
        ax.set_xlabel("Temps (mois)")
        ax.set_ylabel("Probabilité de survie")
        st.pyplot(fig)

    else:
        st.error("❌ Les colonnes 'Time_to_Event' et 'Event_Observed' sont manquantes dans le fichier CSV.")

elif selected_section == "Modèle de régression de Cox":
    st.subheader("📉 Modèle de régression de Cox")

    # 1. Colonnes nécessaires
    required_columns = ["Time_to_Event", "Event_Observed", "Sex", "Tranche_Age",
                        "Smoker", "Treatment", "Physical_Activity"]

    # 2. Vérification des colonnes
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
        st.stop()

    # 3. Création du dataframe de travail
    df_cox = df[required_columns].copy()


    # 4. Conversion robuste des variables catégorielles
    def safe_convert(series, mapping):
        """Fonction helper pour conversion sécurisée"""
        try:
            # Conversion en string et nettoyage
            cleaned = series.astype(str).str.strip().str.title()
            # Application du mapping
            mapped = cleaned.map(mapping)
            # Vérification des valeurs non mappées
            if mapped.isna().any():
                st.warning(f"Valeurs non mappées dans {series.name}: {cleaned[mapped.isna()].unique()}")
                mapped = mapped.fillna(mapping.get('default', 0))
            return mapped.astype(int)
        except Exception as e:
            st.error(f"Erreur conversion {series.name}: {str(e)}")
            st.write("Valeurs uniques:", series.unique())
            st.stop()


    # Mappings complets avec valeurs par défaut
    mappings = {
        'Sex': {'Male': 0, 'Female': 1, 'M': 0, 'F': 1, 'default': 0},
        'Smoker': {'No': 0, 'Yes': 1, '0': 0, '1': 1, 'Non': 0, 'Oui': 1, 'default': 0},
        'Treatment': {'Standard': 0, 'Experimental': 1, 'default': 0},
        'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2, 'default': 1},
        'Tranche_Age': {'<50': 0, '50-60': 1, '>60': 2, 'default': 1}
    }

    # Application des conversions
    for col, mapping in mappings.items():
        df_cox[col] = safe_convert(df_cox[col], mapping)

    # 5. Conversion des variables continues
    try:
        df_cox["Time_to_Event"] = pd.to_numeric(df_cox["Time_to_Event"], errors='raise')
        df_cox["Event_Observed"] = pd.to_numeric(df_cox["Event_Observed"], errors='raise')
    except Exception as e:
        st.error(f"Erreur conversion variable continue: {str(e)}")
        st.stop()

    # 6. Vérification finale des types
    st.write("✅ Types des variables après conversion:")
    st.write(df_cox.dtypes)

    # 7. Exécution du modèle
    with st.spinner("Ajustement du modèle en cours..."):
        try:
            cph = CoxPHFitter()
            cph.fit(df_cox,
                    duration_col="Time_to_Event",
                    event_col="Event_Observed",
                    show_progress=True)

            # Affichage des résultats
            st.success("Modèle ajusté avec succès!")

            # Onglets pour résultats
            tab1, tab2, tab3 = st.tabs(["Résumé", "Hazard Ratios", "Visualisation"])

            with tab1:
                st.subheader("Résumé du modèle")
                try:
                    summary = cph.print_summary()

                    if summary is None:
                        st.info("") # "Le résumé du modèle n'est pas disponible."
                    elif isinstance(summary, pd.DataFrame):
                        st.dataframe(summary)  # Display DataFrame directly
                    elif isinstance(summary, str):
                        st.text(summary)  # Preserve formatting
                    else:
                        st.write("Type de résumé inconnu. Affichage brut :")
                        st.write(summary)  # Fallback

                    # --- Detailed Interpretation ---
                    st.markdown("#### Interprétation des coefficients")
                    if isinstance(cph.summary, pd.DataFrame):  # Check if summary is available as DataFrame
                        for var in cph.summary.index:
                            coef = cph.summary.loc[var, 'coef']
                            hr = np.exp(coef)
                            p_value = cph.summary.loc[var, 'p']

                            st.write(f"**Variable:** `{var}`")
                            st.write(f"  - Coefficient (log HR): {coef:.3f}")
                            st.write(f"  - Hazard Ratio (HR): {hr:.3f}")
                            st.write(f"  - p-value: {p_value:.3f}")

                            if p_value < 0.05:
                                st.success(f"  La variable `{var}` est statistiquement significative.")
                            else:
                                st.warning(f"  La variable `{var}` n'est pas statistiquement significative.")

                            if hr > 1:
                                st.write(f"  Un HR de {hr:.3f} suggère un risque accru de l'événement.")
                            elif hr < 1:
                                st.write(f"  Un HR de {hr:.3f} suggère un risque réduit de l'événement.")
                            else:
                                st.write(f"  Un HR de {hr:.3f} suggère aucun effet sur le risque.")

                    else:
                        st.info(
                            "L'interprétation détaillée n'est pas disponible car le résumé du modèle n'est pas au format DataFrame.")


                except Exception as e:
                    st.error(f"Erreur lors de l'affichage du résumé : {e}")

            with tab2:
                st.subheader("Hazard Ratios")
                st.dataframe(cph.hazard_ratios_.sort_values(ascending=False))
                st.write("Interprétation: Un ratio > 1 indique un risque accru, < 1 un risque réduit.")
                st.write(cph.confidence_intervals_)  # Add confidence intervals

                # Interpretation of Hazard Ratios
                st.markdown("""
                **Interprétation des Hazard Ratios :**

                -   Un Hazard Ratio (HR) de 1 indique que la variable n'a aucun effet sur le risque.
                -   Un HR supérieur à 1 suggère un risque accru de l'événement.
                -   Un HR inférieur à 1 suggère un risque réduit de l'événement.

                Par exemple, si le HR pour 'Smoker' est de 1.5, cela signifie que les fumeurs ont un risque 50% plus élevé de l'événement par rapport aux non-fumeurs, en supposant que toutes les autres variables restent constantes.
                """)

            with tab3:
                st.subheader("Visualisation des effets")
                var = st.selectbox("Variable à analyser",
                                   [c for c in df_cox.columns if c not in ['Time_to_Event', 'Event_Observed']])

                try:
                    if df_cox[var].nunique() <= 5:  # Pour variables catégorielles
                        fig, ax = plt.subplots(figsize=(10, 6))
                        values = sorted(df_cox[var].unique())
                        cph.plot_covariate_groups(var, values=values, ax=ax)
                        ax.set_title(f"Effet de {var} sur la survie")
                        st.pyplot(fig)
                    else:  # Pour variables continues
                        fig, ax = plt.subplots(figsize=(10, 6))
                        cph.plot_partial_effects_on_outcome(var, values=[df_cox[var].min(),
                                                                         df_cox[var].median(),
                                                                         df_cox[var].max()], ax=ax)
                        ax.set_title(f"Effet marginal de {var}")
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Visualisation avancée impossible: {str(e)}")
                    # Solution de repli
                    st.write("🔍 Coefficients pour cette variable:")
                    if var in cph.hazard_ratios_.index:
                        st.metric(f"Hazard Ratio pour {var}",
                                  value=round(cph.hazard_ratios_[var], 2),
                                  help="HR > 1 = risque accru, HR < 1 = risque réduit")
                    else:
                        st.write("Variable non trouvée dans le modèle")

                # Prédictions interactives
                st.subheader("Simulateur de survie")
                col1, col2 = st.columns(2)
                with col1:
                    value = st.slider(f"Valeur de {var}",
                                      float(df_cox[var].min()),
                                      float(df_cox[var].max()),
                                      float(df_cox[var].median()))

                with col2:
                    time_points = st.slider("Période d'évaluation",
                                            int(df_cox["Time_to_Event"].min()),
                                            int(df_cox["Time_to_Event"].max()),
                                            int(df_cox["Time_to_Event"].median()))

                # Calcul des prédictions
                pred_df = df_cox.mean().to_frame().T
                pred_df[var] = value

                try:
                    survival_prob = cph.predict_survival_function(pred_df, times=[time_points]).iloc[0, 0]
                    st.metric(f"Probabilité de survie à {time_points} ",
                              value=f"{survival_prob:.1%}",
                              help="Probabilité qu'un patient avec ces caractéristiques survive jusqu'à ce moment")

                    # Courbe de survie complète
                    fig, ax = plt.subplots(figsize=(10, 4))
                    cph.predict_survival_function(pred_df).plot(ax=ax)
                    ax.axvline(x=time_points, color='red', linestyle='--')
                    ax.set_title("Courbe de survie prédite")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Erreur de prédiction: {str(e)}")

                # Vérification de l'hypothèse de proportionnalité
                st.subheader("Vérification de l'hypothèse de proportionnalité des risques")
                try:
                    results = cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)

                    st.write("Résultats de la vérification :")

                    # Improved display of results
                    if results is None:
                        st.info("Aucun test de proportionnalité des risques n'a été effectué.")
                    elif isinstance(results, dict):
                        for var, result in results.items():
                            st.write(f"Variable: {var}")
                            if isinstance(result, bool):
                                st.write(f"  Proportional Hazards: {result}")
                            elif isinstance(result, pd.Series):
                                st.write("  Test Results:")
                                st.dataframe(result.to_frame(name="p-value"))  # Display as DataFrame
                            else:
                                st.write(f"  Résultat: {result}")  # Generic display
                    elif isinstance(results, list):
                        if not results:
                            st.info("Tous les tests de proportionnalité des risques sont passés.")
                        else:
                            for item in results:
                                st.write(f"  {item}")  # Display each item in the list
                    else:
                        st.write(results)  # Fallback to displaying the raw results

                    # Interpretation (as before, but adjust based on displayed 'results')
                    if isinstance(results, dict) and results.get('non-proportional'):
                        non_prop_vars = results['non-proportional']
                        st.warning(f"L'hypothèse de proportionnalité n'est pas vérifiée pour : {non_prop_vars}")
                        st.write(
                            "Interprétation : Les effets de ces variables peuvent varier avec le temps. Le modèle de Cox standard peut ne pas être entièrement approprié.")
                    elif isinstance(results, list) and results:  # Check if the list is not empty
                        non_prop_vars = [item[0] for item in results if item[1] is False]
                        st.warning(f"L'hypothèse de proportionnalité n'est pas vérifiée pour : {non_prop_vars}")
                        st.write(
                            "Interprétation : Les effets de ces variables peuvent varier avec le temps. Le modèle de Cox standard peut ne pas être entièrement approprié.")
                    else:
                        st.success("L'hypothèse de proportionnalité des risques est globalement vérifiée.")

                except Exception as e:
                    st.error(f"Erreur lors de la vérification des hypothèses : {e}")

        except Exception as e:
            st.error(f"Échec de l'ajustement: {str(e)}")
            st.write("🔍 Détails des données utilisées:")
            st.dataframe(df_cox.describe(include='all'))
            st.write("Types finaux:")
            st.write(df_cox.dtypes)

elif selected_section == "Résumé interactif":
    st.title("📊 Résumé - Vue d'ensemble des patients")

    # 🎛️ Filtres dans la sidebar
    st.sidebar.subheader("🎛️ Filtres pour le résumé")
    sex_filter = st.sidebar.multiselect("Sexe :", df["Sex"].dropna().unique(),
                                        default=list(df["Sex"].dropna().unique()))
    smoker_filter = st.sidebar.multiselect("Fumeur :", df["Smoker"].dropna().unique(),
                                           default=list(df["Smoker"].dropna().unique()))
    treat_filter = st.sidebar.multiselect("Traitement :", df["Treatment"].dropna().unique(),
                                          default=list(df["Treatment"].dropna().unique()))
    activity_filter = st.sidebar.multiselect("Activité physique :", df["Physical_Activity"].dropna().unique(),
                                             default=list(df["Physical_Activity"].dropna().unique()))

    df_filtered = df[
        (df["Sex"].isin(sex_filter)) &
        (df["Smoker"].isin(smoker_filter)) &
        (df["Treatment"].isin(treat_filter)) &
        (df["Physical_Activity"].isin(activity_filter))
        ]

    # ✅ Statistiques globales
    st.subheader("🧮 Statistiques générales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients filtrés", len(df_filtered))
    col2.metric("Événements observés", int(df_filtered["Event_Observed"].sum()))
    col3.metric("Durée moyenne", round(df_filtered["Time_to_Event"].mean(), 1))

    # 🔁 Ratios H/F et par classe BMI
    col4, col5 = st.columns(2)
    if "Sex" in df_filtered.columns:
        sex_ratio = df_filtered["Sex"].value_counts(normalize=True) * 100
        with col4:
            st.markdown("#### 🔹 Répartition H/F")
            for k, v in sex_ratio.items():
                st.write(f"{k} : {v:.1f}%")
    if "Tranche_BMI" in df_filtered.columns:
        bmi_ratio = df_filtered["Tranche_BMI"].value_counts(normalize=True) * 100
        with col5:
            st.markdown("#### 🔹 Répartition par classe BMI")
            for k, v in bmi_ratio.items():
                st.write(f"{k} : {v:.1f}%")

    st.markdown("---")

    # 📦 Barplots des variables catégorielles
    st.subheader("📦 Répartition des variables clés")
    col6, col7, col8 = st.columns(3)

    with col6:
        treat_counts = df_filtered["Treatment"].value_counts().reset_index()
        treat_counts.columns = ["Traitement", "count"]
        fig = px.bar(treat_counts, x="Traitement", y="count", title="Type de traitement")
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        smoker_counts = df_filtered["Smoker"].value_counts().reset_index()
        smoker_counts.columns = ["Fumeur", "count"]
        fig = px.bar(smoker_counts, x="Fumeur", y="count", title="Statut tabagique")
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        activity_counts = df_filtered["Physical_Activity"].value_counts().reset_index()
        activity_counts.columns = ["Activité", "count"]
        fig = px.bar(activity_counts, x="Activité", y="count", title="Activité physique")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 📈 Courbe de survie Kaplan-Meier
    st.subheader("📈 Survie estimée (Kaplan-Meier)")
    if "Time_to_Event" in df_filtered.columns and "Event_Observed" in df_filtered.columns:
        kmf = KaplanMeierFitter()
        kmf.fit(df_filtered["Time_to_Event"], event_observed=df_filtered["Event_Observed"])
        fig_km, ax_km = plt.subplots(figsize=(6, 4))
        kmf.plot_survival_function(ax=ax_km)
        ax_km.set_title("Courbe de survie globale")
        ax_km.set_xlabel("Temps")
        ax_km.set_ylabel("Probabilité de survie")
        st.pyplot(fig_km)

        st.markdown(f"""
        🧠 **Interprétation automatique** :
        - Probabilité de survie initiale : 1.0 (100%)
        - Probabilité de survie à {df_filtered["Time_to_Event"].median():.0f} ≈ **{kmf.predict(df_filtered["Time_to_Event"].median()):.2f}**
        - La survie diminue progressivement, reflétant les événements observés dans le temps.
        """)

    st.markdown("---")

    # 📉 Risque cumulé (Nelson-Aalen)
    st.subheader("📉 Courbe du risque cumulé (Nelson-Aalen)")
    naf = NelsonAalenFitter()
    naf.fit(df_filtered["Time_to_Event"], event_observed=df_filtered["Event_Observed"])
    fig_naf, ax_naf = plt.subplots(figsize=(6, 4))
    naf.plot_cumulative_hazard(ax=ax_naf)
    ax_naf.set_title("Risque cumulé au fil du temps")
    ax_naf.set_xlabel("Temps")
    ax_naf.set_ylabel("Risque cumulé")
    st.pyplot(fig_naf)

    st.markdown("""
    🧠 **Interprétation** : le risque cumulé augmente avec le temps, ce qui indique une probabilité croissante que l'événement étudié se produise. Cela permet de mieux comprendre l’évolution du danger au fil de temps.
    """)

    st.markdown("---")

    # 📊 Répartition démographique
    st.subheader("🧓 Répartition démographique")
    col9, col10 = st.columns(2)

    with col9:
        fig_age = px.histogram(df_filtered, x="Age", nbins=20, title="Distribution des âges")
        st.plotly_chart(fig_age, use_container_width=True)

    with col10:
        if "BMI" in df_filtered.columns:
            fig_bmi = px.histogram(df_filtered, x="BMI", nbins=20, title="Distribution du BMI")
            st.plotly_chart(fig_bmi, use_container_width=True)

    st.markdown("---")

    # 📌 Heatmap des corrélations
    st.subheader("📌 Corrélations entre variables numériques")
    numeric_df = df_filtered.select_dtypes(include="number")
    if not numeric_df.empty and len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Carte des corrélations")
        st.pyplot(fig_corr)
    else:
        st.warning("Pas assez de variables numériques pour afficher une carte de corrélation.")

    st.markdown("---")

    # 🧾 Tableau des données filtrées
    st.subheader("📋 Données filtrées")
    st.dataframe(df_filtered.head(100))  # Affiche les 100 premières lignes max
    st.markdown(f"**{len(df_filtered)} lignes affichées**")

elif selected_section == "À propos du projet":
    st.title("À propos de ce projet d'analyse de survie")

    st.markdown("""
        ## 🎓 Projet d'Ingénierie des Données - Analyse de Survie


        ### 👨‍🔬 Contexte :
          Ce projet vise à analyser la survie de patients en fonction de divers facteurs cliniques en utilisant les techniques d'analyse de survie.


        ### 🧪 Méthodologie :
        Les méthodes d'analyse de survie utilisées comprennent :
            - Estimation de Kaplan-Meier
            - Modèle de Cox à risques proportionnels
            - Estimation de Nelson-Aalen


        ### 📦 Technologies :
            - Python
            - Lifelines (pour l'analyse de survie)
            - Streamlit (pour l'application web)
            - Pandas (pour la manipulation des données)
            - Matplotlib et Plotly (pour la visualisation)


        ### 👥 Réalisé par :
            - Rafiq MAHROUG
            - Ornelia FRANCISCO
            - Abdellatif BENBAHA


        ### 🔗 Liens utiles :
            - [Documentation Lifelines](https://lifelines.readthedocs.io/)
            - [Documentation Streamlit](https://docs.streamlit.io/)
    """)
