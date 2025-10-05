# 🌊 Application de Prévisions Économiques

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**Une application intuitive et pratique pour l'analyse et la prévision de séries temporelles économiques**

[Accéder à l'application](https://rama-forecast.streamlit.app/) • [Signaler un bug](https://github.com/ramanambonona/forecast) • [Documentation complète](#documentation)

</div>

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Démarrage rapide](#-démarrage-rapide)
- [Guide d'utilisation](#-guide-dutilisation)
- [Modèles disponibles](#-modèles-disponibles)
- [Formats de données](#-formats-de-données)
- [Captures d'écran](#-captures-décran)
- [FAQ](#-faq)
- [Support](#-support)

---

## 🎯 Vue d'ensemble

Cette application web permet d'**analyser, visualiser et prévoir** des données économiques de manière intuitive. 
Conçue pour les économistes, analystes financiers et décideurs, elle offre une interface simple masquant la complexité des modèles statistiques avancés.

### Pourquoi cette application ?

- **Simplicité** : Interface claire et intuitive, aucune compétence en programmation requise
- **Puissance** : 10 modèles de prévision avancés (ARIMA, Prophet, Random Forest, etc.)
- **Flexibilité** : Supporte Excel et CSV avec détection automatique de format
- **Visualisation** : Graphiques interactifs avec zoom, capture et export
- **Analyse automatique** : Recommandations de modèles basées sur vos données

---

## ⚡ Fonctionnalités

### 📥 Module de Collecte des Données

- **Import facile** : Glissez-déposez vos fichiers Excel (.xlsx, .xls) ou CSV
- **Auto-détection** : L'application détecte automatiquement l'orientation de vos données
- **Nettoyage intelligent** : Conversion automatique des dates et valeurs numériques
- **Validation** : Vérification de la qualité des données avant traitement

### 📊 Module de Visualisation

**Indicateurs clés**
- Affichage des métriques principales avec évolution (base année de départ)
- Formatage automatique en milliers (k), millions (M), milliards (Md)

**Types de graphiques**
- Courbes d'évolution temporelle
- Barres verticales et horizontales
- Graphiques en aire
- Histogrammes de distribution
- Box plots statistiques

**Statistiques descriptives**
- Moyenne, médiane, écart-type
- Minimum, maximum, quartiles
- Analyse exploratoire complète

### 🔍 Module d'Analyse

**Analyse automatique des séries temporelles**
- Détection de tendance (forte, modérée, faible)
- Identification de saisonnalité (forte, modérée, absente)
- Recommandations de modèles adaptés

**Décomposition de séries**
- Composante tendancielle
- Composante saisonnière
- Composante résiduelle
- Visualisation interactive multi-courbes

### 📈 Module de Prévisions

**10 modèles disponibles**
- SSAE (Simple Seasonal Average Estimation)
- AR(p) - Autorégressif
- ARIMA - AutoRegressive Integrated Moving Average
- VAR - Vector AutoRegression
- ARDL - AutoRegressive Distributed Lag
- Prophet (Facebook)
- Régression Linéaire
- Random Forest
- MLP (Multi-Layer Perceptron)
- Exponential Smoothing

**Fonctionnalités avancées**
- Paramétrage personnalisé pour chaque modèle
- Horizon de prévision ajustable (3 à 60 mois)
- Calcul automatique de la précision (MAPE)
- Export Excel individuel ou complet
- Graphiques avec distinction historique/prévision

### 🖼️ Interaction avec les graphiques

Chaque graphique dispose d'une **barre d'outils interactive** :

- **📷 Capture** : Téléchargez le graphique en PNG haute résolution
- **🔍 Zoom** : Sélectionnez une zone pour zoomer
- **↔️ Panoramique** : Déplacez la vue
- **🏠 Réinitialiser** : Retour à la vue initiale
- **📏 Autoscale** : Ajustement automatique des axes

---

## 🚀 Démarrage rapide

### En ligne (recommandé)

1. Accédez à l'application : [https://rama-forecast.streamlit.app/](https://rama-forecast.streamlit.app/)
2. Importez votre fichier de données
3. Lancez votre première prévision !

### En local

```bash
# Cloner le repository
git clone https://github.com/ramanambonona/forecast-app.git
cd forecast-app

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run main-prev-V1-fixed.py
```

**Prérequis**
- Python 3.8+
- pip (gestionnaire de packages Python)

---

## 📖 Guide d'utilisation

### Étape 1 : Importation des données

1. Cliquez sur **"📥 Data"** dans la barre latérale
2. Glissez-déposez votre fichier ou cliquez pour parcourir
3. Sélectionnez l'orientation des données (ou laissez en auto-détection)
4. Ajustez les options si nécessaire (lignes/colonnes à ignorer)
5. Cliquez sur **"Valider et sauvegarder"**

### Étape 2 : Exploration des données

1. Cliquez sur **"📈 Prév."** dans la barre latérale
2. Consultez les **Indicateurs Clés** en haut de page
3. Explorez l'onglet **"Évolution"** pour voir les tendances
4. Utilisez l'onglet **"Visualisation"** pour des graphiques avancés

### Étape 3 : Analyse des séries

1. Allez dans l'onglet **"Analyse"**
2. Sélectionnez une variable à analyser
3. Consultez les caractéristiques détectées (tendance, saisonnalité)
4. Notez les **recommandations de modèles**

### Étape 4 : Prévisions

1. Accédez à l'onglet **"Prévisions"**
2. Choisissez un **modèle** (suivez les recommandations de l'analyse)
3. Sélectionnez l'**indicateur** à prévoir
4. Définissez l'**horizon de prévision** (en mois)
5. Ajustez les **paramètres** du modèle si nécessaire
6. Cliquez sur **"Lancer la prévision"**

### Étape 5 : Export des résultats

**Pour une seule variable :**
- Cliquez sur **"Exporter les prévisions (unique)"**
- Téléchargez le fichier Excel

**Pour toutes les variables :**
- Cliquez sur **"Générer Excel avec toutes les prévisions"**
- Téléchargez le fichier complet

**Pour les graphiques :**
- Survolez le graphique
- Cliquez sur l'icône 📷 dans la barre d'outils
- L'image PNG est téléchargée automatiquement

---

## 🧮 Modèles disponibles

### Modèles simples

**SSAE - Simple Seasonal Average Estimation**
- Prévision basée sur la moyenne mobile
- Idéal pour : Données stables sans tendance marquée
- Paramètres : Aucun

**AR(p) - AutoRegressive**
- Prévision basée sur les valeurs passées
- Idéal pour : Données avec autocorrélation
- Paramètres : Ordre p (nombre de retards)

**Régression Linéaire**
- Modèle de tendance linéaire simple
- Idéal pour : Tendances linéaires claires
- Paramètres : Aucun

### Modèles ARIMA

**ARIMA - AutoRegressive Integrated Moving Average**
- Modèle classique de séries temporelles
- Idéal pour : Données avec tendance, peut gérer la saisonnalité
- Paramètres : 
  - p (ordre autorégressif)
  - d (ordre de différenciation)
  - q (ordre moyenne mobile)

**ARDL - AutoRegressive Distributed Lag**
- Extension d'ARIMA avec variables exogènes
- Idéal pour : Relations de causalité entre variables
- Paramètres : Nombre de retards

**VAR - Vector AutoRegression**
- Modèle multivarié pour variables interdépendantes
- Idéal pour : Analyse de plusieurs variables corrélées
- Paramètres : Ordre des retards
- Note : Utilise automatiquement 2 variables minimum

### Modèles avancés

**Prophet (Facebook)**
- Modèle spécialisé pour séries avec forte saisonnalité
- Idéal pour : Données avec saisonnalité et changements de tendance
- Paramètres :
  - Échelle prior changepoint (flexibilité de la tendance)
  - Échelle prior saisonnalité (force de la saisonnalité)

**Random Forest**
- Ensemble d'arbres de décision
- Idéal pour : Relations non-linéaires complexes
- Paramètres :
  - Nombre d'estimateurs (arbres)
  - Profondeur maximale

**MLP - Multi-Layer Perceptron**
- Réseau de neurones artificiels
- Idéal pour : Patterns complexes, grandes quantités de données
- Paramètres :
  - Tailles des couches cachées
  - Nombre d'itérations

**Exponential Smoothing**
- Lissage exponentiel avec tendance et saisonnalité
- Idéal pour : Données avec composantes multiples
- Paramètres :
  - Type de tendance (additive/multiplicative)
  - Type de saisonnalité (additive/multiplicative)
  - Période saisonnière

---

## 📁 Formats de données

### Structure attendue

Votre fichier doit contenir :
- **Une colonne de dates** (première colonne ou détectée automatiquement)
- **Des colonnes de variables numériques** (une par indicateur)

### Formats de dates supportés

```
2024-01-15        (YYYY-MM-DD)
15-01-2024        (DD-MM-YYYY)
01-15-2024        (MM-DD-YYYY)
Jan 2024          (Mois YYYY)
janvier 2024      (Mois français)
2024M01           (YYYY M MM)
202401            (YYYYMM)
```

### Exemple de structure

**Option 1 : Variables en colonnes (recommandé)**

| Date | PIB | Inflation | Chômage |
|------|-----|-----------|---------|
| 2020-01 | 1500000 | 2.1 | 8.5 |
| 2020-02 | 1520000 | 2.3 | 8.3 |
| 2020-03 | 1510000 | 2.5 | 8.7 |

**Option 2 : Variables en lignes**

| Variable | 2020-01 | 2020-02 | 2020-03 |
|----------|---------|---------|---------|
| PIB | 1500000 | 1520000 | 1510000 |
| Inflation | 2.1 | 2.3 | 2.5 |
| Chômage | 8.5 | 8.3 | 8.7 |

### Conseils de préparation

✅ **À faire**
- Utilisez des noms de colonnes clairs
- Évitez les cellules fusionnées
- Supprimez les lignes de totaux/moyennes
- Gardez un format de date cohérent
- Utilisez des points (.) pour les décimales

❌ **À éviter**
- Plusieurs feuilles dans un fichier Excel (seule la première sera lue)
- Des formules Excel (seules les valeurs sont importées)
- Des caractères spéciaux dans les noms de variables
- Des lignes vides au milieu des données

---

## 📸 Captures d'écran

### Module de collecte
```
┌──────────────────────────────────────┐
│  📥 IMPORTATION                      │
│  [Glisser-déposer ou Parcourir]     │
│                                      │
│  ✓ Auto-détection du format         │
│  ✓ Aperçu des données               │
│  ✓ Validation avant traitement      │
└──────────────────────────────────────┘
```

### Indicateurs clés
```
┌─────────┬─────────┬─────────┬─────────┐
│   PIB   │  Infla  │ Chômage │ Exports │
│ 2,5 Md  │  3,2 M  │  125 k  │ 450 M   │
│ +12.5%  │  +2.1%  │  -5.3%  │ +8.9%   │
└─────────┴─────────┴─────────┴─────────┘
```

### Graphique de prévision
```
     Valeur
       │
       │     ──── Historique
       │    ╱
       │   ╱
       │  ╱
       │ ╱ ┊ ┄┄┄ Prévision
       │╱  ┊  ┄┄
       └───┊────────> Temps
           │
     Début prévision
```

---

## ❓ FAQ

### Questions générales

**Q : L'application est-elle gratuite ?**  
R : Oui, l'application est totalement gratuite et open-source.

**Q : Mes données sont-elles sécurisées ?**  
R : Oui, toutes les données sont traitées localement dans votre session et ne sont jamais stockées sur nos serveurs.

**Q : Puis-je utiliser l'application hors ligne ?**  
R : Non, l'application nécessite une connexion internet. Cependant, vous pouvez l'installer en local (voir section Démarrage rapide).

### Questions techniques

**Q : Combien de données faut-il minimum ?**  
R : 
- Minimum absolu : 12 points de données
- Recommandé : 24+ points pour l'analyse de saisonnalité
- Optimal : 36+ points pour les modèles complexes

**Q : Pourquoi mon modèle retourne une erreur ?**  
R : Vérifiez que :
- Vous avez suffisamment de données pour le modèle choisi
- Vos données ne contiennent pas trop de valeurs manquantes
- Les paramètres sont adaptés à la taille de votre série

**Q : Comment choisir le bon modèle ?**  
R : 
1. Lancez l'analyse automatique (onglet "Analyse")
2. Consultez les recommandations basées sur vos données
3. Testez plusieurs modèles et comparez les MAPE

**Q : Que signifie le MAPE ?**  
R : Le MAPE (Mean Absolute Percentage Error) mesure la précision du modèle :
- < 10% : Excellente précision
- 10-20% : Bonne précision
- 20-50% : Précision acceptable
- > 50% : Précision faible, essayez un autre modèle

### Problèmes courants

**Q : Mes dates ne sont pas reconnues**  
R : 
- Vérifiez que la colonne de dates est en première position
- Utilisez un format standard (voir section Formats de données)
- Évitez les formats de date personnalisés Excel

**Q : Les graphiques ne s'affichent pas**  
R : 
- Actualisez la page (F5)
- Vérifiez votre connexion internet
- Essayez avec un autre navigateur (Chrome recommandé)

**Q : L'export Excel ne fonctionne pas**  
R : 
- Vérifiez que votre navigateur autorise les téléchargements
- Essayez avec un autre navigateur
- Désactivez temporairement les bloqueurs de publicités

---

## 💡 Astuces et bonnes pratiques

### Pour de meilleures prévisions

1. **Nettoyez vos données** avant l'import
   - Supprimez les lignes/colonnes inutiles
   - Corrigez les valeurs aberrantes
   - Comblez les valeurs manquantes si possible

2. **Commencez simple**
   - Testez d'abord les modèles simples (SSAE, AR, Régression)
   - Passez aux modèles complexes si nécessaire

3. **Utilisez l'analyse automatique**
   - Les recommandations sont basées sur les caractéristiques de vos données
   - Suivez les suggestions de modèles

4. **Comparez plusieurs modèles**
   - Testez 2-3 modèles différents
   - Comparez les MAPE
   - Choisissez le modèle le plus précis

5. **Ajustez l'horizon de prévision**
   - Plus l'horizon est long, moins la prévision est fiable
   - Privilégiez des horizons courts (3-12 mois)

### Pour une meilleure visualisation

1. **Limitez le nombre de variables affichées**
   - 3-4 variables maximum par graphique
   - Créez plusieurs graphiques si nécessaire

2. **Utilisez le bon type de graphique**
   - Lignes : pour les évolutions temporelles
   - Barres : pour les comparaisons
   - Box plots : pour les distributions

3. **Interagissez avec les graphiques**
   - Zoomez sur les périodes intéressantes
   - Capturez les graphiques importants
   - Utilisez le reset pour revenir à la vue initiale

---

## 🛠️ Support

### Besoin d'aide ?

- **Documentation** : Consultez ce README en détail
- **Issues GitHub** : [Signaler un bug](https://github.com/ramanambonona/forecast)
- **Email** : ambinintsoa.uat.ead2@gmail.com

### Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Streamlit** pour le framework web
- **Plotly** pour les graphiques interactifs
- **statsmodels** pour les modèles statistiques
- **Prophet** (Meta) pour le modèle de prévision
- **scikit-learn** pour les modèles de machine learning

---
---

**Ramanambonona Ambinintsoa, Ph.D**

| [![Mail](https://img.icons8.com/?size=30&id=86875&format=png&color=000000)](mailto:ambinintsoa.uat.ead2@gmail.com) | [![GitHub](https://img.icons8.com/?size=30&id=3tC9EQumUAuq&format=png&color=000000)](https://github.com/ramanambonona) | [![LinkedIn](https://img.icons8.com/?size=30&id=8808&format=png&color=000000)](https://www.linkedin.com/in/ambinintsoa-ramanambonona) |
| :---: | :---: | :---: |

---
<div align="center">
[⬆ Retour en haut](#-application-de-prévisions-économiques)

</div>
