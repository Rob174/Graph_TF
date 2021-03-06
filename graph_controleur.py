import graph_layer
from graph_layer import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D, Concatenate, Input
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
def custom_accuracy(y_true,y_pred):
    return K.mean(K.equal(y_true,y_pred))
compteur_model = 0
trop_param = False
class G_Controleur:
    prec_echec = True
    def __init__(self,hparam):
        self.couche_id = 0
        self.couches_graph = []
        self.nb_liens = 0
        self.hp = hparam
        self.graph = Digraph("graph",format='png')
        self.nb_couches = {
            "conv":{"min":4,"max":20,"choix":None},
            "deconv":{"min":0,"max":5,"choix":None},
            "pool":{"min":0,"max":5,"choix":None},
            "add":{"min":0,"max":30,"choix":None},
            "dense":{"min":0,"max":10,"choix":None}
        }
        liste_classes = [G_Conv,G_Deconv,G_Pool,G_Add,G_Dense]
        G_Input(self)
        # Crée le nb de couches choisies par Keras Tuner et retourne le nb créé
        for name,v,Class in zip(self.nb_couches.items(),liste_classes):
            v["choix"] = self.hp.Int("nb_"+name,min_value=v["min"],max_value=v["max"])
            for _ in range(v["choix"]):
                Class(self)
        G_Output(self)
        self.couches_graph[0].eval()
        self.lier(0,1,forcer=True)
        for n in self.couches_graph:
            n.link()
        self.afficher("post_liaison")
        #On fournit l'input aux noeuds racines
        for n in self.couches_graph:
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
        for n in self.couches_graph:
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
                loss='MSE',metrics=[custom_accuracy])
        os.system("cp -r /content/Bayesian_optimization '/content/drive/My Drive'")
        max_param = -1
        if os.path.isfile("/content/Graph_TF/max_param.txt") == True:
            with open("/content/Graph_TF/max_param.txt","r") as f:
                for i,l in enumerate(f):
                    if i == 0:
                        max_param = int(l.strip())
                    elif i  == 1:
                        prec_echec = l.strip()
                        retour_ancienne_exec = int(prec_echec)
                        max_param = retour_ancienne_exec if (max_param==-1 and max_param < retour_ancienne_exec) and G_Controleur.prec_echec == True else max_param
                        G_Controleur.prec_echec = False
        
        if (self.model.count_params() >= max_param and max_param !=-1):
            print("Trop de parametre avec %d pour ce modèle et précédement échec avec %d (-1 = infini)"%(self.model.count_params(),max_param))
            global trop_param
            trop_param = True
            return
        with open("/content/Graph_TF/max_param.txt","w") as f:
            f.write(str(max_param)+"\n")
            f.write(str(self.model.count_params())+"\n")
    def clean(self):
        del self.couche_id
        del self.nb_liens
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
        print("%d noeuds et %d liens contre %d au max soit %f prct et %d lien par noeud en moyenne"%(len(self.couches_graph),self.nb_liens,len(self.couches_graph)*(len(self.couches_graph)-1),10**-2*int(10000*self.nb_liens/(len(self.couches_graph)*(len(self.couches_graph)-1))),self.nb_liens/len(self.couches_graph)))
        self.graph.render("./graph_%s"%name)
    def lier(self,couche_id_1,couche_id_2,forcer=False,adapt=False):
        if adapt == False and ((self.couches_graph[couche_id_1].invisible_adapt == True  and "Output" not in self.couches_graph[couche_id_1].__class__.__name__) or (self.couches_graph[couche_id_2].invisible_adapt == True and "Output" not in self.couches_graph[couche_id_2].__class__.__name__)):
            return
        lien = False
        #Vérifications
        verifications_ok = couche_id_1 != couche_id_2 and "Output" not in self.couches_graph[couche_id_1].__class__.__name__#Ce n'est pas la couche courante
        verifications_ok = verifications_ok and "Input" not in self.couches_graph[couche_id_2].__class__.__name__
        verifications_ok = verifications_ok and couche_id_2 not in self.couches_graph[couche_id_1].parents#Ce n'est pas un parent de la couche mère
        if verifications_ok == False:
            return
        #Calcul de la difference de taille
        #nb de réduction de taille (pool = -1 ; deconv = +1) : dimension < 0 => réduit globalement la taille
        
        tailles_source,parents_1 = self.couches_graph[couche_id_1].get_size_parent_list_parents()
        tailles_dest_enfants,enfants_2 = self.couches_graph[couche_id_2].get_size_enfant_list_enfants()
        tailles_dest_parents,parents_2 = self.couches_graph[couche_id_2].get_size_parent_list_parents()
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
        self.couches_graph[couche_id_2].tmp_parents = self.couches_graph[couche_id_2].parents
        self.couches_graph[couche_id_2].tmp_parents.append(self.couches_graph[couche_id_1].couche_id)
        self.couches_graph[couche_id_2].tmp_parents = list(dict.fromkeys(self.couches_graph[couche_id_2].parents))
        verif_boucle = self.couches_graph[couche_id_2].test_actualiser_enfant(couche_id_1)
        if verif_boucle == False:
            return
        #Vérifie si les tailles sont compatibles
        verification_taille = False
        if len(self.couches_graph[couche_id_1].parent) == 0:
            verification_taille = True
        elif len(self.couches_graph[couche_id_2].parent) == 0 and len(self.couches_graph[couche_id_2].enfant)==0:
            verification_taille = True
        else:
            diff_taille = tailles_source[0]-(tailles_dest_parents[0]+self.couches_graph[couche_id_2].couche_pool-self.couches_graph[couche_id_2].couche_deconv)#Calcul la différence entre la taille à la sortie de la première couche et celle à l'entrée!! de la seconde
            if diff_taille == 0:
                verification_taille = True
            else:
                couche_adapt = self.couches_graph[couche_id_1]
                if diff_taille > 0:
                    for i in range(diff_taille):
                        adapt = graph_layer.G_Pool(self)
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id,forcer=True,adapt=True)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                else:
                    for i in range(-diff_taille):
                        adapt = graph_layer.G_Deconv(self)
                        self.couches_graph[adapt.couche_id].invisible_adapt = True
                        self.lier(couche_adapt.couche_id,self.couches_graph[adapt.couche_id].couche_id,forcer=True,adapt=True)
                        couche_adapt = self.couches_graph[adapt.couche_id]
                self.lier(self.couches_graph[couche_adapt.couche_id].couche_id,self.couches_graph[couche_id_2].couche_id,adapt=True,forcer=True)
                return 
        test_boucle = self.couches_graph[couche_id_2].test_actualiser_enfant(couche_id_1)
        if test_boucle == False:
            return
        choix_lien = False
        if verification_taille == True and forcer == False:
            choix_lien = self.hp.Choice("lien_%s%d_%s%d"%(self.couches_graph[couche_id_1].__class__.__name__,self.couches_graph[couche_id_1].couche_id_type,self.couches_graph[couche_id_2].__class__.__name__,self.couches_graph[couche_id_2].couche_id_type),[True,False],default=False) 
        else:
            choix_lien = True
        if couche_id_2 == 3:
            break_pt2 = -2
        if couche_id_1 == 4 and couche_id_2 == 1:
            self.afficher("breakpoint")
            break_pt = -1
        if verification_taille==True and choix_lien == True:
            self.nb_liens += 1
            self.graph.edge(str(self.couches_graph[couche_id_1].couche_id),str(self.couches_graph[couche_id_2].couche_id))
            self.couches_graph[couche_id_2].parent.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents.append(self.couches_graph[couche_id_1].couche_id)
            self.couches_graph[couche_id_2].parents = list(dict.fromkeys(self.couches_graph[couche_id_2].parents))
            self.couches_graph[couche_id_2].actualiser_enfant()
            self.couches_graph[couche_id_1].enfant.append(self.couches_graph[couche_id_2].couche_id)
def create_model(hparam):
    G_Conv.compteur = 0
    G_Deconv.compteur = 0
    G_Pool.compteur = 0
    G_Conv.compteur = 0
    G_Dense.compteur = 0
    G_Add.compteur = 0
    global compteur_model
    compteur_model += 1
    controleur = G_Controleur(hparam)
    model = controleur.model
    print("Nb param : ",model.count_params())
    controleur.clean()
    del controleur
    global trop_param
    if trop_param == True:
        inpt = Input(shape=(256,256,3),dtype=tf.dtypes.float32,name='Entree_env10x256x256x3')
        outpt = tf.keras.layers.Subtract()([inpt,inpt])
        outpt = tf.keras.layers.GaussianNoise(5)(outpt)
        try:
            del model
        except:
            pass
        model = Model(inputs=inpt,outputs=outpt,name='Model_invalide')
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1,
                                                momentum=0,
                                                nesterov=False),
                    loss='MSE',metrics=[custom_accuracy])
        trop_param = False
    return model