from G_Layer import G_Layer
from G_Conv import G_Conv
from G_Deconv import G_Deconv
from G_Input import G_Input
from G_Pool import G_Pool
from HP_test import HP_test
from graphviz import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.losses
import tensorflow as tf
hp = HP_test()
nb_max_layer = 200
possibilites_filtres = [1,3,10,50,100,500]








print("Début")
G_Layer.couches_graph = []
G_Layer.couche_id = 0
Lcouches = []
nb_pool = 0
#Création des couches
nb_couches = 50#hp.Int("nb_layers",min_value=1,max_value=nb_max_layer)
G_Input(hp)
for _ in range(nb_couches):
    couches_autorisees = ['conv','deconv','pool']
    type_couche = hp.Choice('choix_couche_%d'%(G_Layer.couche_id),couches_autorisees,default='conv')
    if type_couche == 'conv':
        G_Conv()
    elif type_couche == 'deconv':
        G_Deconv()
    elif type_couche == 'pool':
        G_Pool()
#Liaisons entre les couches
G_Layer.couches_graph[0].eval()
G_Layer.couches_graph[0].lier(G_Layer.couches_graph[1].couche_id,forcer=True)
for n in G_Layer.couches_graph:
    n.link()
G_Layer.afficher("post_liaison")
#On fournit l'input aux noeuds racines
print('Nb_noeuds : ',len(G_Layer.couches_graph))
for i,n in enumerate(G_Layer.couches_graph):
    if len(n.parent)==0:
        if i==1:
            print("class %s, id %d"%(n.__class__.__name__,n.couche_id))
        G_Layer.couches_graph[0].lier(n.couche_id,forcer=True)
        n.couche_input = G_Layer.couches_graph[0].couche
    print("parents après input de %s: "%n.__class__.__name__,n.parent)
G_Layer.afficher("ajout_input")
#Evaluation des couches possibles (connaissant leur entrée)
def r(nodes):
    print("step")
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
            return r(nodes)
r(G_Layer.couches_graph[:])
#Liaison à la fin
Llayer_fin = []
couche_fin = None
for l in G_Layer.couches_graph:
    if l.enfant == []:#Pour toutes les couches feuilles de l'arbre
        dim_noeud = -l.nb_pool_parent_tot+     l.nb_deconv_parent_tot
        couche = l.couche_output
        if dim_noeud > 0:
            while dim_noeud != 0:
                couche = MaxPooling2D(pool_size=2,strides=2,padding='VALID',name='MaxPool_id_%d_fin_layer_%d'%(l.couche_id,dim_noeud))(couche)
                dim_noeud -= 1
        elif dim_noeud < 0:
            while dim_noeud != 0:
                couche = Conv2DTranspose(filters=filters,kernel_size=2,strides=2,name='TransposedConv_Deconv_id_%d_adapt_layer_%d'%(self.couche_id,abs(dim_noeud)))(couche)
                dim_noeud += 1
        
        Llayer_fin.append(couche)
if len(Llayer_fin) > 1:
    couche_fin = Concatenate(axis=-1)(Llayer_fin)
else:
    couche_fin = Llayer_fin[0]
couche_fin = Conv2D(filters=3,
                            kernel_size=2,
                            padding='SAME',
                            activation='linear',
                            name='Convolution_fin_k2_f3')(couche_fin)
    
model = Model(inputs=G_Layer.couches_graph[0].couche,outputs=couche_fin,name='Debruiteur')
print("Nb param : ",model.count_params())
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Choice('lr',[1.,0.1,0.01,0.001,10**-4,10**-5],default=0.01),
                                                momentum=hp.Choice('momentum',[1.,0.1,0.01,0.001,10**-4,10**-5,0.],default=0),
                                                nesterov=False),
                loss='MSE',metrics=[])#custom_accuracy_fct(model.count_params(),7e4),'accuracy'])