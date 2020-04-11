#Graph_TF
Le but de ce projet est d'automatiser entièrement la construction de modèle et de voir si certains éléments ressortent
## Formulation 1 (dernier commit 11 avril 2020)
### Implémentation

Choisi les couches avec au maximum 10 couches de chaque : Conv2D, MaxPooling2D, AveragePooling2D (ces deux derniers choisis ensembles cf implémentation), Deconvolution + éventuellement d'autres couches si on veut lier deux couches pour lesquelle les tailles d'images sont différentes. 

Dans l'implémentation, pour faciliter la gestion de la taille des images, on se limitera à des couches de Pooling et de Déconvolution avec un noyau de 2.

Le reste des hyperparamètres sera choisi par une optimisation Bayesienne du module Keras Tuner.

### Résultats

La session d'entrainement s'arrête au bout d'un moment car la mémoire est saturée (avec 3 997 420 paramètres)
Précision maximale : cf Modele000002_Keras_Bayesian_libre --> 0.337

## Formulation 1.1
### Implémentation (à faire)

Ajouter dans la valeur de précision utilisée par Keras Tuner une fonction décroissante du nombre de paramètre, négative qui fait tendre la précision vers 0 quand il y a trop de paramètres

### Résultats

## Formulation 2
### Implémentation (à faire)

Ajouter les couches de dropout, batchnormalization et regularisation

### Résultats