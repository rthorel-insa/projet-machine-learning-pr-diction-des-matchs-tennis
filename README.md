# ATP Match Predictor - Machine Learning 🎾

Ce projet utilise l'apprentissage automatique (Random Forest) pour prédire l'issue des matchs de tennis professionnels de l'ATP. Le modèle s'appuie sur des données historiques pour calculer des statistiques dynamiques (forme, ratio par surface, face-à-face) avant chaque match.

## 📊 Présentation du projet
L'objectif est de transformer les données brutes des matchs de l'ATP en variables explicatives (features) pertinentes pour entraîner une IA capable de déterminer les probabilités de victoire entre deux joueurs.
Ce projet a été réalisé avec l'aide de l'IA Gemini.

### Variables clés utilisées (Features) :
* **Classement ATP** : Prise en compte du rang et de la différence de niveau entre les joueurs.
* **Forme Récente** : Calculée sur les 5 derniers matchs via une file d'attente (deque).
* **Spécialisation par Surface** : Ratios de victoire spécifiques (Terre battue, Dur, Gazon).
* **Indice Mental** : Performance spécifique dans les moments clés (Grands Chelems, Finales et Demi-finales).
* **Head-to-Head (H2H)** : Historique des confrontations directes entre les deux joueurs.

---

## 🚀 Installation et Dépendances

1. **Cloner le projet** :
   ```bash
   git clone [https://github.com/votre-utilisateur/atp-predictor.git](https://github.com/votre-utilisateur/atp-predictor.git)
