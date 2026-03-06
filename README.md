# Apprentissage Hybride sur Puce Quantique JPC (Josephson Parametric Converter)

Ce projet de recherche, mené dans le cadre d'un partenariat entre l'**École Polytechnique** et **Thales**, explore l'utilisation de systèmes quantiques supraconducteurs pour des tâches de traitement du signal et de classification via des techniques d'apprentissage automatique hybride.

---

## 1. Présentation du Projet

L'objectif est de simuler et d'optimiser les paramètres d'une puce quantique de type **JPC (Josephson Parametric Converter)** pour extraire des caractéristiques (*features*) de signaux temporels. Le système combine :
* **Une couche quantique (Black Box) :** Une simulation physique basée sur l'Hamiltonien d'un circuit JPC utilisant la bibliothèque `dynamiqs`.
* **Une optimisation d'ordre zéro (Zeroth-Order) :** Utilisée pour ajuster les paramètres de la puce quantique (comme les constantes de couplage $g_{conv}$ et $g_{sq}$) sans nécessiter de gradient analytique du système physique.
* **Une couche classique :** Un réseau de neurones classique (optimisé par Adam) qui classifie les sorties traitées par la puce quantique.

---

## 2. Architecture du Code

Le projet est structuré comme suit dans le répertoire `quantum/` :

* **`jpc_chip.py`** : Cœur de la simulation physique. Définit les opérateurs de création/annihilation, l'Hamiltonien de Kerr, les drives et les opérateurs de dissipation pour le JPC. Il gère également l'intégration temporelle via `dq.mesolve`.
* **`quantum_black_box.py`** : Interface faisant le pont entre la simulation physique et l'algorithme d'optimisation. Elle permet d'effectuer des passages avant (*forward*) perturbés pour l'estimation du gradient.
* **`quantummodelconfig.py`** : Orchestrateur de l'entraînement hybride. Il gère la boucle d'apprentissage, alternant entre l'optimisation des paramètres quantiques (Zeroth-Order) et des poids du réseau de neurones (First-Order).
* **`data.py`** : Générateur de données synthétiques. Le projet utilise actuellement une tâche de classification binaire : différencier un signal **sinusoïdal** d'un signal **carré**.
* **`configs.py`** : Configuration des hyperparamètres (taux d'apprentissage, échelles de perturbation, dimensions des couches).
* **`main.py`** : Point d'entrée pour lancer l'instanciation du modèle et l'entraînement.

---

## 3. Modèle Physique : Le JPC

La simulation repose sur un système à deux modes ($a$ et $b$) dont l'Hamiltonien inclut :
* Des termes de Kerr ($K_{AA}, K_{BB}$) et de Kerr croisé ($K_{AB}$).
* Des termes de couplage paramétrique ($g_{conv}, g_{sq}$) qui sont les paramètres cibles de l'apprentissage.
* Des drives externes proportionnels au signal d'entrée $x(t)$.

Les sorties de la puce sont récupérées sous forme de quadratures ($I$ et $Q$) pour chaque mode, constituant le vecteur de caractéristiques final injecté dans le classifieur classique.

---

## 4. Installation et Dépendances

Le projet nécessite les bibliothèques suivantes :
* `numpy` & `pandas` : Traitement des données et gestion des courbes.
* `jax` & `dynamiqs` : Simulation haute performance de la dynamique quantique.
* `matplotlib` : Visualisation des résultats et de la perte (*loss*).
* **`zeroth`** : Une bibliothèque interne dédiée aux optimiseurs d'ordre zéro.

---

## 5. Utilisation

Pour lancer l'entraînement du modèle avec les configurations par défaut définies dans `main.py` :

```bash
python -m quantum