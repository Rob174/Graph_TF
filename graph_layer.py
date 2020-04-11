from graphviz import *
from graphviz import Digraph
import numpy as np
from tensorflow.keras.layers import Concatenate

possibilites_filtres = [1,3,10,50,100,500]
class G_Layer:
    def __init__(self,controleur):
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
        self.controleur = controleur
        self.couche_id = self.controleur.new_id()
        self.couche_id_type = -1
        self.parent = []#parent(s) direct(s) index
        self.parents = []
        self.tmp_parents = []
        self.enfant = []#directes uniquement                            ******************************Appliquer la fonction d'update et voir si assez d'avoir uniquement les fils directs
        self.taille = None #taille de l'image manipulée
        self.couche_pool = 0#couche courante couche de pooling
        self.couche_deconv = 0#couche courante couche de deconv
        self.couche_input = None
        self.couche_output = None
        self.invisible_adapt = False

        #Debug graphviz
    def clean(self):
        del self.controleur
        del self.couche_id
        del self.couche_id_type
        del self.parent
        del self.parents
        del self.tmp_parents
        del self.enfant
        del self.taille
        del self.couche_pool
        del self.couche_deconv
        del self.invisible_adapt
    def test_actualiser_enfant(self,new_id):
        ok = True
        self.controleur.couches_graph[e].tmp_parents.append(new_id)
        self.controleur.couches_graph[e].tmp_parents += self.controleur.couches_graph[new_id].parents
        self.controleur.couches_graph[e].tmp_parents = list(dict.fromkeys(self.controleur.couches_graph[e].tmp_parents))
        for e in self.enfant:
            self.controleur.couches_graph[e].parents += self.parents
            self.controleur.couches_graph[e].parents = list(dict.fromkeys(self.controleur.couches_graph[e].parents))
            ok = self.controleur.couches_graph[e].test_actualiser_enfant(new_id)
            if e in self.controleur.couches_graph[e].parents:
                ok = False
        return ok

    def actualiser_enfant(self):
        for e in self.enfant:
            self.controleur.couches_graph[e].parents += self.parents
            self.controleur.couches_graph[e].parents = list(dict.fromkeys(self.controleur.couches_graph[e].parents))
            self.controleur.couches_graph[e].actualiser_enfant()
            if e in self.controleur.couches_graph[e].parents:
                raise Exception("Boucle")
    def get_size_parent(self):
        if "Output" in self.__class__.__name__:
            return [0]
        if len(self.parent) == 0:
            return [self.couche_deconv-self.couche_pool]
        else:
            tailles = []
            for p in self.parent:
                size = self.controleur.couches_graph[p].get_size_parent()
                tailles += size
            tailles = list(np.array(tailles)+self.couche_deconv-self.couche_pool)
            return tailles
    def get_size_enfant(self):
        if "Output" in self.__class__.__name__:
            return [0]
        if len(self.enfant) == 0:
            return [self.couche_deconv-self.couche_pool]
        else:
            tailles = []
            for p in self.enfant:
                size = self.controleur.couches_graph[p].get_size_enfant()
                tailles += size
            tailles = list(np.array(tailles)+self.couche_deconv-self.couche_pool)
            return tailles

    def link(self):
        #Uniformisation du nb de couches de pooling
        #On choisit si on va lier la couche courante vers la couche fille ...
        if self.invisible_adapt == False:
            for i,couche in enumerate(self.controleur.couches_graph):
                if couche.invisible_adapt == False:
                    self.controleur.lier(self.couche_id,i)
    def eval(self):
            if self.parent != [] and True not in list(map(lambda x:self.controleur.couches_graph[x].couche_output == None,self.parent)):
                if len(self.parent) > 1:
                    L = [self.controleur.couches_graph[i].couche_output for i in self.parent]
                    self.couche_output = self.couche(Concatenate(axis=-1)(L)) if "Output" not in self.__class__.__name__ else Concatenate(axis=-1)(L)
                else:
                    self.couche_output = self.couche(self.controleur.couches_graph[self.parent[0]].couche_output) if "Output" not in self.__class__.__name__ else self.controleur.couches_graph[self.parent[0]].couche_output
                return True
            return False

from tensorflow.keras.layers import Conv2D

class G_Conv(G_Layer):
    compteur = 0
    def __init__(self,controleur):
        super(G_Conv,self).__init__(controleur)
        self.couche_id_type = G_Conv.compteur
        G_Conv.compteur += 1
        self.filters = self.controleur.hp.Choice('filtre_conv_index_%d'%(self.couche_id_type),possibilites_filtres,default=1)
        self.kernel = self.controleur.hp.Choice('kernel_conv_index_%d'%(self.couche_id_type),[2,3],default=2)
        self.activ = self.controleur.hp.Choice("activation_conv_index_%d"%(self.couche_id_type),['linear','relu','elu','selu','tanh'],default='relu')
        self.couche = Conv2D(filters=self.filters,
                             kernel_size=self.kernel,
                             padding='SAME',
                             activation=self.activ,
                             name='Convolution_id_gen_%d_k%d_f%d_activ_%s'%(self.controleur.couche_id,self.filters,self.kernel,self.activ))
        self.controleur.add_couche(self)
        self.controleur.graph.node(str(self.couche_id),shape='record',label="{Conv %d-%d|{{Noyau|%d}|{Filtres|%d}|{Activation|%s}}}"%(self.couche_id,self.couche_id_type,self.kernel,self.filters,self.activ))
    def clean(self):
        super(G_Conv,self).clean()
        del self.filters
        del self.kernel
        del self.activ

from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
class G_Pool(G_Layer):
    compteur = 0
    def __init__(self,controleur):
        super(G_Pool,self).__init__(controleur)
        self.couche_id_type = G_Pool.compteur
        G_Pool.compteur += 1
        self.type_pool = self.controleur.hp.Choice('pool_index_%d'%(self.couche_id_type),['avg','max'],default='max')
        self.couche_pool = 1
        if self.type_pool == 'max':
            self.couche = MaxPooling2D(pool_size=2,strides=2,padding='VALID',name='MaxPool_id_%d'%(self.couche_id))
        else:
            self.couche = AveragePooling2D(pool_size=2,strides=2,padding='VALID',name='AveragePool_id_%d'%(self.couche_id))
        self.controleur.add_couche(self)
        self.controleur.graph.node(str(self.couche_id),shape='record',label="{%spooling %d-%d}"%("Max" if self.type_pool == 'max' else 'Average',self.couche_id,self.couche_id_type))
    def clean(self):
        super(G_Pool,self).clean()
        del self.type_pool

from tensorflow.keras.layers import Input
import tensorflow as tf
class G_Input(G_Layer):
    def __init__(self,controleur):
        super(G_Input,self).__init__(controleur)
        
        self.couche=Input(shape=(256,256,3),dtype=tf.dtypes.float32,name='Entree_env10x256x256x3')
        self.controleur.add_couche(self)
        self.controleur.graph.node(str(self.couche_id),shape='record',label="{Input %d}"%(self.couche_id))
    def eval(self):
        self.couche_output = self.couche
    def link(self):
        pass

from tensorflow.keras.layers import Conv2DTranspose
class G_Deconv(G_Layer):
    compteur = 0
    def __init__(self,controleur):
        super(G_Deconv,self).__init__(controleur)
        self.couche_id_type = G_Deconv.compteur
        G_Deconv.compteur += 1
        self.filters = self.controleur.hp.Choice('filtre_conv_index_%d'%(self.couche_id_type),possibilites_filtres,default=1)
        self.activ = self.controleur.hp.Choice("activation_conv_index_%d"%(self.couche_id_type),['linear','relu','elu','selu','tanh'],default='relu')
        self.couche = Conv2DTranspose(filters=self.filters,kernel_size=2,strides=2,name='TransposedConv_Deconv_id_%d_f%d'%(self.couche_id,self.filters))
        self.couche_deconv = 1
        self.controleur.add_couche(self)
        self.controleur.graph.node(str(self.couche_id),shape='record',label="{Deconv %d-%d|{{Filtres|%d}|{Activation|%s}}}"%(self.couche_id,self.couche_id_type,self.filters,self.activ))
    def clean(self):
        super(G_Deconv,self).clean()
        del self.filters
        del self.activ

class G_Output(G_Layer):
    def __init__(self,controleur):
        super(G_Output,self).__init__(controleur)
        self.invisible_adapt = True
        self.controleur.add_couche(self)
        self.controleur.graph.node(str(self.couche_id),shape='record',label="Output %d"%self.couche_id)
        
