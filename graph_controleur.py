import graph_layer
from graph_layer import *
from HP_test import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.losses
import tensorflow as tf
import os
from graphviz import *
from graphviz import Digraph
class G_Controleur:
    def __init__(self):
        self.couche_id = 0
        self.couches_graph = []
        self.hp = HP_test()
        self.graph = Digraph("graph",format='png')
        self.nb_conv = self.hp.Int("nb_conv",min_value=0,max_value=100)
        self.nb_deconv = self.hp.Int("nb_deconv",min_value=0,max_value=100)
        self.nb_pool = self.hp.Int("nb_pool",min_value=0,max_value=100)
        G_Input(self.hp)
        for i in range(self.nb_conv):
            G_Conv(self.hp)
        for i in range(self.nb_deconv):
            G_Deconv(self.hp)
        for i in range(self.nb_pool):
            G_Pool(self.hp)
        G_Output(self.hp)
        self.couches_graph[0].eval()
        self.lier(0,1,forcer=True)
        for n in self.couches_graph:
            n.link()
        self.afficher("post_liaison")
        #On fournit l'input aux noeuds racines
        for i,n in enumerate(G_Controleur.couches_graph):
            if len(n.parent)==0:
                G_Controleur.lier(G_Controleur.couches_graph[0].couche_id,n.couche_id,forcer=True)
                n.couche_input = G_Controleur.couches_graph[0].couche
        G_Controleur.afficher("ajout_input")
        # On lie à l'output
        i=len(G_Controleur.couches_graph)-1
        while G_Controleur.couches_graph[i].__class__.__name__ != "G_Output" and i >= 0:
            i -= 1
        if i < 0:
            raise Exception("Output missing")
        for j,n in enumerate(G_Controleur.couches_graph):
            if len(n.enfant) == 0 and n.__class__.__name__ != "G_Output":
                G_Controleur.lier(n.couche_id,G_Controleur.couches_graph[i].couche_id,forcer=True)
        G_Controleur.afficher("fin_liaisons")
        # On évalue les couches qui peuvent l'être
        couches_graph_eval = r(self.couches_graph[:])
        # On remet avec une dernière couche à la bonne taille
        couche_fin = Conv2D(filters=3,kernel_size=2,padding='SAME',activation='linear',name='conv_fin_k2_f3')(couches_graph_fin[index_output].output_couche)
        #On affiche les graphs
        dossier_fichier = os.path.abspath(os.path.dirname(__file__))
        for img in os.listdir(dossier_fichier):
            if img.endswith(".png"):
                os.system("display ./"+img)
        # On construit le modèle
        self.model = Model(inputs=couches_graph_eval.output_couche,outputs=couche_fin,name='Debruiteur')
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Choice('lr',[1.,0.1,0.01,0.001,10**-4,10**-5],default=0.01),
                                                momentum=hp.Choice('momentum',[1.,0.1,0.01,0.001,10**-4,10**-5,0.],default=0),
                                                nesterov=False),
                loss='MSE',metrics=["accuracy"])
    def eval_rec(self,nodes):
        if len(nodes)==0:
            return
        else:
            index = 0 
            while index < len(nodes) and nodes[index].eval() == False:
                index += 1
            if index == len(nodes):
                for n in nodes:
                    if n.couche_input == None:
                        print("Noeud %d : parents directes : "%n.couche_id,n.parent)
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
        if (couche_id_1 == 11 or couche_id_1 == 12) and couche_id_2 == 1:
            point_arret = 0
        lien = False
        #Vérifications
        verifications_ok = couche_id_1 != couche_id_2 and "Output" not in self.couches_graph[couche_id_1].__class__.__name__#Ce n'est pas la couche courante
        verifications_ok = verifications_ok and "Input" not in self.couches_graph[couche_id_2].__class__.__name__
        verifications_ok = verifications_ok and couche_id_2 not in self.couches_graph[couche_id_1].parents#Ce n'est pas un parent de la couche mère
        if verifications_ok == False:
            print("Impossible de lier %d et %d"%(couche_id_1,couche_id_2))
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
        #Vérifie si les tailles sont compatibles
        verification_taille = False
        taille_2 = None
        if len(self.couches_graph[couche_id_2].parent) == 0:
            verification_taille = True
        else:
            diff_taille = tailles_source[0]-(tailles_dest_parents[0]+self.couches_graph[couche_id_2].couche_pool-self.couches_graph[couche_id_2].couche_deconv)
            if diff_taille == 0:
                verification_taille = True
            elif forcer == True:
                couche_adapt = self.couches_graph[couche_id_1]
                if diff_taille > 0:
                    for i in range(diff_taille):
                        adapt = graph_layer.G_Pool()
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                else:
                    for i in range(-diff_taille):
                        adapt = graph_layer.G_Deconv()
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                self.lier(self.couches_graph[couche_adapt.couche_id].couche_id,self.couches_graph[couche_id_2].couche_id)
                return 
                
        if verification_taille==True:
            print("Lien entre %d et %d"%(couche_id_1,couche_id_2))
            self.graph.edge(str(self.couches_graph[couche_id_1].couche_id),str(self.couches_graph[couche_id_2].couche_id))
            self.couches_graph[couche_id_2].parent.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents = list(dict.fromkeys(self.couches_graph[couche_id_2].parents))
            for p in self.couches_graph[couche_id_2].parents:
                if p not in self.couches_graph[couche_id_2].parents:
                    self.couches_graph[couche_id_2].parents.append(p)
            self.couches_graph[couche_id_1].enfant.append(self.couches_graph[couche_id_2].couche_id)

