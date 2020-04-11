import graph_layer
from graph_layer import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.losses
import tensorflow as tf
import os
from graphviz import *
from graphviz import Digraph
os.system("pip install -U keras-tuner") #De https://github.com/keras-team/keras-tuner
from kerastuner import BayesianOptimization, Objective
#from tensorboard.plugins.hparams import api as hp

compteur_model = 0
class G_Controleur:
    def __init__(self,hparam):
        self.couche_id = 0
        self.couches_graph = []
        self.hp = hparam
        self.graph = Digraph("graph",format='png')
        self.nb_conv = self.hp.Int("nb_conv",min_value=4,max_value=10)
        self.nb_deconv = self.hp.Int("nb_deconv",min_value=0,max_value=10)
        self.nb_pool = self.hp.Int("nb_pool",min_value=0,max_value=10)
        G_Input(self)
        for i in range(self.nb_conv):
            G_Conv(self)
        for i in range(self.nb_deconv):
            G_Deconv(self)
        for i in range(self.nb_pool):
            G_Pool(self)
        G_Output(self)
        self.couches_graph[0].eval()
        self.lier(0,1,forcer=True)
        for i,n in enumerate(self.couches_graph):
            n.link()
        self.afficher("post_liaison")
        #On fournit l'input aux noeuds racines
        for i,n in enumerate(self.couches_graph):
            if len(n.parent)==0:
                print("Liaison théorique entre %d et %d"%(self.couches_graph[0].couche_id,n.couche_id))
                self.lier(self.couches_graph[0].couche_id,n.couche_id,forcer=True)
                n.couche_input = self.couches_graph[0].couche
        self.afficher("ajout_input")
        # On lie à l'output
        index_output=len(self.couches_graph)-1
        while self.couches_graph[index_output].__class__.__name__ != "G_Output" and i >= 0:
            index_output -= 1
        if index_output < 0:
            raise Exception("Output missing")
        for j,n in enumerate(self.couches_graph):
            if len(n.enfant) == 0 and n.__class__.__name__ != "G_Output":
                self.lier(n.couche_id,self.couches_graph[index_output].couche_id,forcer=True)
        self.afficher("fin_liaisons")
        # On évalue les couches qui peuvent l'être
        self.eval_rec(self.couches_graph[:])
        # On remet avec une dernière couche à la bonne taille
        couche_fin = Conv2D(filters=3,kernel_size=2,padding='SAME',activation='linear',name='conv_fin_k2_f3')(self.couches_graph[index_output].couche_output)
        #On affiche les graphs
        dossier_fichier = os.path.abspath(os.path.dirname(__file__))
        for img in os.listdir(dossier_fichier):
            if img.endswith(".png"):
                os.system("display ./"+img)
        # On construit le modèle
        self.model = Model(inputs=self.couches_graph[0].couche_output,outputs=couche_fin,name='Debruiteur')
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.hp.Choice('lr',[1.,0.1,0.01,0.001,10**-4,10**-5],default=0.01),
                                                momentum=self.hp.Choice('momentum',[1.,0.1,0.01,0.001,10**-4,10**-5,0.],default=0),
                                                nesterov=False),
                loss='MSE',metrics=["accuracy"])
    def clean(self):
        del self.couche_id
        for c in self.couches_graph:
            c.clean()
        del self.graph
        
    def eval_rec(self,nodes):
        
        if len(nodes)==0:
            return
        else:
            index = 0 
            while index < len(nodes) and nodes[index].eval() == False:
                index += 1
            if index == len(nodes):
                raise Exception("Modèle impossible")
            else:
                nodes.pop(index)
                return self.eval_rec(nodes)    
    def new_id(self):
        self.couche_id += 1
        return self.couche_id-1
    def add_couche(self,couche):
        self.couches_graph.append(couche)
    def afficher(self,name):
        self.graph.render("./graph_%s"%name)
    def lier(self,couche_id_1,couche_id_2,forcer=False):
        lien = False
        #Vérifications
        verifications_ok = couche_id_1 != couche_id_2 and "Output" not in self.couches_graph[couche_id_1].__class__.__name__#Ce n'est pas la couche courante
        verifications_ok = verifications_ok and "Input" not in self.couches_graph[couche_id_2].__class__.__name__
        verifications_ok = verifications_ok and couche_id_2 not in self.couches_graph[couche_id_1].parents#Ce n'est pas un parent de la couche mère
        if verifications_ok == False:
            return
        #Calcul de la difference de taille
        #nb de réduction de taille (pool = -1 ; deconv = +1) : dimension < 0 => réduit globalement la taille
        
        tailles_source = self.couches_graph[couche_id_1].get_size_parent()
        tailles_dest_enfants = self.couches_graph[couche_id_2].get_size_enfant()
        tailles_dest_parents = self.couches_graph[couche_id_2].get_size_parent()
        #Le chemin jusqu'à la racine côté couche courante et le chemin jusqu'aux feuilles n'a pas trop de couche de pooling au total
        #Si les dimensions de sortie du layer courant et celles d'entrée du layer cibles coincident
        taille_si_lie = 0
        if len(tailles_source) != 0:
            taille_si_lie += min(tailles_source)
        if len(tailles_dest_enfants) != 0:
            taille_si_lie += min(tailles_dest_enfants)
        
        if taille_si_lie < -8:#Si on a trop de couches de pooling si on liait les deux couches
            lier = False
            return
        verif_boucle = self.couches_graph[couche_id_2].test_actualiser_enfant(couche_id_1)
        if verif_boucle == False:
            return
        #Vérifie si les tailles sont compatibles
        verification_taille = False
        #taille_2 = None
        if len(self.couches_graph[couche_id_1].parent) == 0:
            verification_taille = True
        else:
            diff_taille = tailles_source[0]-(tailles_dest_parents[0]+self.couches_graph[couche_id_2].couche_pool-self.couches_graph[couche_id_2].couche_deconv)
            if diff_taille == 0:
                verification_taille = True
            elif forcer == True:
                couche_adapt = self.couches_graph[couche_id_1]
                if diff_taille > 0:
                    for i in range(diff_taille):
                        adapt = graph_layer.G_Pool(self)
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                else:
                    for i in range(-diff_taille):
                        adapt = graph_layer.G_Deconv(self)
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                self.lier(self.couches_graph[couche_adapt.couche_id].couche_id,self.couches_graph[couche_id_2].couche_id)
                return 
        choix_lien = False
        if verification_taille == True and forcer == False:
            choix_lien = self.hp.Choice("lien_%s%d_%s%d"%(self.couches_graph[couche_id_1].__class__.__name__,self.couches_graph[couche_id_1].couche_id_type,self.couches_graph[couche_id_2].__class__.__name__,self.couches_graph[couche_id_2].couche_id_type),[True,False],default=False) 
        else:
            choix_lien = True
        if verification_taille==True and choix_lien == True:
            self.graph.edge(str(self.couches_graph[couche_id_1].couche_id),str(self.couches_graph[couche_id_2].couche_id))
            self.couches_graph[couche_id_2].parent.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents = list(dict.fromkeys(self.couches_graph[couche_id_2].parents))
            self.couches_graph[couche_id_2].actualiser_enfant()
            self.couches_graph[couche_id_1].enfant.append(self.couches_graph[couche_id_2].couche_id)
def create_model(hparam):
    print("Model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    G_Conv.compteur = 0
    G_Deconv.compteur = 0
    G_Pool.compteur = 0
    G_Conv.compteur = 0
    global compteur_model
    compteur_model += 1
    controleur = G_Controleur(hparam)
    model = controleur.model
    controleur.clean()
    del controleur
    return model