import graph_layer
from graphviz import Digraph
class G_Controleur:
    couche_id = 0
    couches_graph = []
    hp = None
    graph = Digraph("graph",format='png')
    @staticmethod
    def new_id():
        G_Controleur.couche_id += 1
        return G_Controleur.couche_id-1
    @staticmethod
    def add_couche(couche):
        G_Controleur.couches_graph.append(couche)
    @staticmethod
    def afficher(name):
        G_Controleur.graph.render("./graph_%s"%name)
    # Pour l'actualisation des parents faire un système pseudo récursif
    @staticmethod
    def lier(couche_id_1,couche_id_2,forcer=False):
        if (couche_id_1 == 11 or couche_id_1 == 12) and couche_id_2 == 1:
            point_arret = 0
        lien = False
        #Vérifications
        verifications_ok = couche_id_1 != couche_id_2 and "Output" not in G_Controleur.couches_graph[couche_id_1].__class__.__name__#Ce n'est pas la couche courante
        verifications_ok = verifications_ok and "Input" not in G_Controleur.couches_graph[couche_id_2].__class__.__name__
        verifications_ok = verifications_ok and couche_id_2 not in G_Controleur.couches_graph[couche_id_1].parents#Ce n'est pas un parent de la couche mère
        if verifications_ok == False:
            print("Impossible de lier %d et %d"%(couche_id_1,couche_id_2))
            return
        #Calcul de la difference de taille
        #nb de réduction de taille (pool = -1 ; deconv = +1) : dimension < 0 => réduit globalement la taille
        
        #noeud courant, dimensions
        dim_crt =                   -G_Controleur.couches_graph[couche_id_1].nb_pool_parent_tot -G_Controleur.couches_graph[couche_id_1].couche_pool+G_Controleur.couches_graph[couche_id_1].couche_deconv+     G_Controleur.couches_graph[couche_id_1].nb_deconv_parent_tot
        dim_crt_enfant  = dim_crt     -G_Controleur.couches_graph[couche_id_1].nb_pool_enfant_tot+       G_Controleur.couches_graph[couche_id_1].nb_deconv_enfant_tot
        #noeud peut-être enfant, dimensions
        dim_nd_sel =                          -G_Controleur.couches_graph[couche_id_2].nb_pool_parent_tot+   G_Controleur.couches_graph[couche_id_2].nb_deconv_parent_tot    #on exclut la couche courante
        dim_nd_sel_enfant = dim_nd_sel          -G_Controleur.couches_graph[couche_id_2].nb_pool_enfant_tot+     G_Controleur.couches_graph[couche_id_2].nb_deconv_enfant_tot     -G_Controleur.couches_graph[couche_id_2].couche_pool  +G_Controleur.couches_graph[couche_id_2].couche_deconv#on ajoute la couche courante
        #Le chemin jusqu'à la racine côté couche courante et le chemin jusqu'aux feuilles n'a pas trop de couche de pooling au total
        #Si les dimensions de sortie du layer courant et celles d'entrée du layer cibles coincident
        diff_taille = dim_crt-dim_nd_sel
        print("Difference de taille : %d de la couche %d vers %d"%(diff_taille,couche_id_1,couche_id_2))
        if diff_taille == 0:
            if min(dim_crt+dim_crt_enfant,dim_crt+dim_nd_sel_enfant) >= 0 and diff_taille == 0:
                lien = G_Controleur.hp.Choice('lien_layer_%d_vers_%d'%(G_Controleur.couches_graph[couche_id_1].couche_id,G_Controleur.couches_graph[couche_id_2].couche_id),[False,True],default=False)
            if forcer == True:
                lien = True
        if diff_taille != 0 and forcer == True:
            couche_adapt = G_Controleur.couches_graph[couche_id_1]
            if diff_taille > 0:
                for i in range(diff_taille):
                    adapt = graph_layer.G_Pool()
                    G_Controleur.couches_graph[adapt.couche_id].invisible_adapt = True
                    G_Controleur.lier(couche_adapt.couche_id,G_Controleur.couches_graph[adapt.couche_id].couche_id,forcer=True)
                    couche_adapt = G_Controleur.couches_graph[adapt.couche_id]
            else:
                for i in range(-diff_taille):
                    adapt = graph_layer.G_Deconv()
                    G_Controleur.couches_graph[adapt.couche_id].invisible_adapt = True
                    G_Controleur.lier(couche_adapt.couche_id,G_Controleur.couches_graph[adapt.couche_id].couche_id,forcer=True)
                    couche_adapt = G_Controleur.couches_graph[adapt.couche_id]
            G_Controleur.lier(G_Controleur.couches_graph[couche_adapt.couche_id].couche_id,G_Controleur.couches_graph[couche_id_2].couche_id,forcer=True)
            return 
        if lien==True:
            print("Lien entre %d et %d"%(couche_id_1,couche_id_2))
            G_Controleur.graph.edge(str(G_Controleur.couches_graph[couche_id_1].couche_id),str(G_Controleur.couches_graph[couche_id_2].couche_id))
            G_Controleur.couches_graph[couche_id_2].parent.append(G_Controleur.couches_graph[couche_id_1].couche_id)
            G_Controleur.couches_graph[couche_id_2].parents.append(G_Controleur.couches_graph[couche_id_1].couche_id)
            G_Controleur.couches_graph[couche_id_2].parents = list(dict.fromkeys(G_Controleur.couches_graph[couche_id_2].parents))
            for p in G_Controleur.couches_graph[couche_id_2].parents:
                if p not in G_Controleur.couches_graph[couche_id_2].parents:
                    G_Controleur.couches_graph[couche_id_2].parents.append(p)
            G_Controleur.couches_graph[couche_id_1].enfant.append(G_Controleur.couches_graph[couche_id_2].couche_id)

            G_Controleur.couches_graph[couche_id_1].update_max_branches(type_brch='enfant')
            G_Controleur.couches_graph[couche_id_2].update_max_branches(type_brch='parent')

