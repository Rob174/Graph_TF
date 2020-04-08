from graphviz import *
from graphviz import Digraph
import graph_controleur
import numpy as np

possibilites_filtres = [1,3,10,50,100,500]
class G_Layer:
    def __init__(self):
        """
        #Arguments
        - hp : optimiseur Keras-tuner
        #Fonctionnement
        Création d'un G_Layer --> Ajout à la liste des noeuds ; 
        Ajout d'un lien : pour un layer parcourt les layers dans l'ordre. Si on peut lier le layer courant avec celui parcouru on tente de les lier
            on fait des adaptations de taille si besoin avec des couches de pooling ou de convolution 
        Une fois tous les liens faits on devra lier toutes les couches de sortie dans un concatenate en veillant à ce qu'elles aient des mêmes dimensions
        On devra également fornir aux couches racines (sans parent) l'input
        """
        self.couche_id = graph_controleur.G_Controleur.new_id()
        self.couche_id_type = -1
        self.parent = []#parent(s) direct(s) index
        self.parents = []
        self.enfant = []#directes uniquement                            ******************************Appliquer la fonction d'update et voir si assez d'avoir uniquement les fils directs
        self.taille = None #taille de l'image manipulée
        self.couche_pool = 0#couche courante couche de pooling
        self.couche_deconv = 0#couche courante couche de deconv
        self.couche_input = None
        self.couche_output = None
        self.invisible_adapt = False

        #Debug graphviz
    def get_size_parent(self):
        if len(self.parent) == 0:
            return []
        else:
            tailles = []
            for p in self.parent:
                size = graph_controleur.G_Controleur.couches_graph[p].get_size_parent()
                if size == []:
                    size = [0]
                tailles += size
            tailles = list(np.array(tailles)+self.couche_deconv-self.couche_pool)
            return tailles
    def get_size_enfant(self):
        if len(self.enfant) == 0:
            return []
        else:
            tailles = []
            for p in self.enfant:
                size = graph_controleur.G_Controleur.couches_graph[p].get_size_enfant()
                if size == []:
                    size = [0]
                tailles += size
            tailles = list(np.array(tailles)+self.couche_deconv-self.couche_pool)
            return tailles

    def update_max_branches(self,type_brch='parents'):
        Lnb_pool = None
        Lnb_deconv = None
        liste_concernee = None
        if type_brch == 'parent':
            liste_concernee = self.parent
            Lnb_pool = [0 for i in range(len(self.parent))]
            Lnb_deconv = [0 for i in range(len(self.parent))]
        else:
            liste_concernee = self.enfant
            Lnb_pool = [0 for i in range(len(self.enfant))]
            Lnb_deconv = [0 for i in range(len(self.enfant))]
        for index,p in enumerate(liste_concernee):
            if type_brch == 'parent':
                Lnb_pool[index] = graph_controleur.G_Controleur.couches_graph[index].nb_pool_parent_tot
                Lnb_deconv[index] = graph_controleur.G_Controleur.couches_graph[index].nb_deconv_parent_tot
            else:
                Lnb_pool[index] = graph_controleur.G_Controleur.couches_graph[index].nb_pool_enfant_tot+self.couche_pool
                Lnb_deconv[index] = graph_controleur.G_Controleur.couches_graph[index].nb_deconv_enfant_tot+self.couche_deconv
        if type_brch == 'parent':
            self.nb_pool_parent_tot = max(Lnb_pool) if len(Lnb_pool) > 0 else 0
            self.nb_deconv_parent_tot = max(Lnb_deconv) if len(Lnb_deconv) > 0 else 0
        else:
            self.nb_pool_enfant_tot = max(Lnb_pool) if len(Lnb_pool) > 0 else 0
            self.nb_deconv_enfant_tot = max(Lnb_deconv) if len(Lnb_deconv) > 0 else 0
    def link(self):
        #Uniformisation du nb de couches de pooling
        #On choisit si on va lier la couche courante vers la couche fille ...
        if self.invisible_adapt == False:
            for i,couche in enumerate(graph_controleur.G_Controleur.couches_graph):
                if couche.invisible_adapt == False:
                    graph_controleur.G_Controleur.lier(self.couche_id,i)
    def eval(self):
            if True not in list(map(lambda x:G_Layer.couches_graph[x].couche_output == None,self.parent)):
                if len(self.parent) > 1:
                    L = [G_Layer.couches_graph[i].couche_output for i in self.parent]
                    self.couche_output = self.couche(Concatenate(axis=-1)(L))
                else:
                    print("parents : ",self.parent)
                    self.couche_output = self.couche(G_Layer.couches_graph[self.parent[0]].couche_output)
                return True
            return False

from tensorflow.keras.layers import Conv2D

class G_Conv(G_Layer):
    compteur = 0
    def __init__(self):
        super(G_Conv,self).__init__()
        self.couche_id_type = G_Conv.compteur
        G_Conv.compteur += 1
        self.filters = graph_controleur.G_Controleur.hp.Choice('filtre_conv_index_%d'%(self.couche_id_type),possibilites_filtres,default=1)
        self.kernel = graph_controleur.G_Controleur.hp.Choice('kernel_conv_index_%d'%(self.couche_id_type),[2,3],default=2)
        self.activ = graph_controleur.G_Controleur.hp.Choice("activation_conv_index_%d"%(self.couche_id_type),['linear','relu','elu','selu','tanh'],default='relu')
        self.couche = Conv2D(filters=self.filters,
                             kernel_size=self.kernel,
                             padding='SAME',
                             activation=self.activ,
                             name='Convolution_id_gen_%d_k%d_f%d_activ_%s'%(graph_controleur.G_Controleur.couche_id,self.filters,self.kernel,self.activ))
        graph_controleur.G_Controleur.add_couche(self)
        graph_controleur.G_Controleur.graph.node(str(self.couche_id),shape='record',label="{Conv %d-%d|{{Noyau|%d}|{Filtres|%d}|{Activation|%s}}}"%(self.couche_id,self.couche_id_type,self.kernel,self.filters,self.activ))

from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
class G_Pool(G_Layer):
    compteur = 0
    def __init__(self):
        super(G_Pool,self).__init__()
        self.couche_id_type = G_Pool.compteur
        G_Pool.compteur += 1
        self.type_pool = graph_controleur.G_Controleur.hp.Choice('pool_index_%d'%(self.couche_id_type),['avg','max'],default='max')
        self.couche_pool = 1
        if self.type_pool == 'max':
            self.couche = MaxPooling2D(pool_size=2,strides=2,padding='VALID',name='MaxPool_id_%d'%(self.couche_id))
        else:
            self.couche = AveragePooling2D(pool_size=2,strides=2,padding='VALID',name='AveragePool_id_%d'%(self.couche_id))
        graph_controleur.G_Controleur.add_couche(self)
        graph_controleur.G_Controleur.graph.node(str(self.couche_id),shape='record',label="{%spooling %d-%d}"%("Max" if self.type_pool == 'max' else 'Average',self.couche_id,self.couche_id_type))

from tensorflow.keras.layers import Input
import tensorflow as tf
class G_Input(G_Layer):
    def __init__(self,hp):
        super(G_Input,self).__init__()
        graph_controleur.G_Controleur.hp = hp
        self.couche=Input(shape=(256,256,3),dtype=tf.dtypes.float32,name='Entree_env10x256x256x3')
        graph_controleur.G_Controleur.add_couche(self)
        graph_controleur.G_Controleur.graph.node(str(self.couche_id),shape='record',label="{Input %d}"%(self.couche_id))
    def eval(self):
        self.couche_output = self.couche
    def link(self):
        pass

from tensorflow.keras.layers import Conv2DTranspose
class G_Deconv(G_Layer):
    compteur = 0
    def __init__(self):
        super(G_Deconv,self).__init__()
        self.couche_id_type = G_Deconv.compteur
        G_Deconv.compteur += 1
        self.filters = graph_controleur.G_Controleur.hp.Choice('filtre_conv_index_%d'%(self.couche_id_type),possibilites_filtres,default=1)
        self.activ = graph_controleur.G_Controleur.hp.Choice("activation_conv_index_%d"%(self.couche_id_type),['linear','relu','elu','selu','tanh'],default='relu')
        self.couche = Conv2DTranspose(filters=self.filters,kernel_size=2,strides=2,name='TransposedConv_Deconv_id_%d_f%d'%(self.couche_id,self.filters))
        self.couche_deconv = 1
        graph_controleur.G_Controleur.add_couche(self)
        graph_controleur.G_Controleur.graph.node(str(self.couche_id),shape='record',label="{Deconv %d-%d|{{Filtres|%d}|{Activation|%s}}}"%(self.couche_id,self.couche_id_type,self.filters,self.activ))

class G_Output(G_Layer):
    def __init__(self):
        super(G_Output,self).__init__()
        self.invisible_adapt = True
        graph_controleur.G_Controleur.add_couche(self)
        graph_controleur.G_Controleur.graph.node(str(self.couche_id),shape='record',label="Output %d"%self.couche_id)
        