from graph_layer import *
from graph_controleur import *
from HP_test import *
from graphviz import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.losses
import tensorflow as tf
import os
hp = HP_test()
nb_max_layer = 3
possibilites_filtres = [1,3,10,50,100,500]
G_Input(hp)
G_Conv()
G_Pool()
G_Conv()
G_Pool()
G_Conv()
G_Pool()
G_Conv()
G_Output()
G_Controleur.couches_graph[0].eval()
G_Controleur.lier(G_Controleur.couches_graph[0].couche_id,G_Controleur.couches_graph[1].couche_id,forcer=True)
for n in G_Controleur.couches_graph:
    n.link()
G_Controleur.afficher("post_liaison")
#On fournit l'input aux noeuds racines
print('Nb_noeuds : ',len(G_Controleur.couches_graph))
for i,n in enumerate(G_Controleur.couches_graph):
    if len(n.parent)==0:
        G_Controleur.lier(G_Controleur.couches_graph[0].couche_id,n.couche_id,forcer=True)
        n.couche_input = G_Controleur.couches_graph[0].couche

        if n.couche_id==2:
            print("2 Après ajout : ",n.parents)
    print("parents après input de %s %d : "%(n.__class__.__name__,n.couche_id),n.parent)
G_Controleur.afficher("ajout_input")
i=len(G_Controleur.couches_graph)-1
while G_Controleur.couches_graph[i].__class__.__name__ != "G_Output" and i >= 0:
    i -= 1
if i < 0:
    raise Exception("Output missing")
for i,n in enumerate(G_Controleur.couches_graph):
    if len(n.enfant) == 0:
        G_Controleur.lier(n.couche_id,G_Controleur.couches_graph[i].couche_id,forcer=True)
G_Controleur.afficher("fin")
dossier_fichier = os.path.abspath(os.path.dirname(__file__))
for img in os.listdir(dossier_fichier):
    if img.endswith(".png"):
        os.system("display ./"+img)
