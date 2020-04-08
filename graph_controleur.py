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
        
        tailles_source = G_Controleur.couches_graph[couche_id_1].get_size_parent()
        tailles_dest_enfants = G_Controleur.couches_graph[couche_id_2].get_size_enfant()
        tailles_dest_parents = G_Controleur.couches_graph[couche_id_2].get_size_parent()
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
        if len(G_Controleur.couches_graph[couche_id_2].parent) == 0:
            verification_taille = True
        else:
            diff_taille = tailles_source[0]-(tailles_dest_parents[0]+G_Controleur.couches_graph[couche_id_2].couche_pool-G_Controleur.couches_graph[couche_id_2].couche_deconv)
            if diff_taille == 0:
                verification_taille = True
            elif forcer == True:
                couche_adapt = G_Controleur.couches_graph[couche_id_1]
                if diff_taille > 0:
                    for i in range(diff_taille):
                        adapt = graph_layer.G_Pool()
                        G_Controleur.couches_graph[adapt.couche_id].invisible_adapt = True
                        G_Controleur.lier(couche_adapt.couche_id,G_Controleur.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = G_Controleur.couches_graph[adapt.couche_id]
                else:
                    for i in range(-diff_taille):
                        adapt = graph_layer.G_Deconv()
                        G_Controleur.couches_graph[adapt.couche_id].invisible_adapt = True
                        G_Controleur.lier(couche_adapt.couche_id,G_Controleur.couches_graph[adapt.couche_id].couche_id)
                        couche_adapt = G_Controleur.couches_graph[adapt.couche_id]
                G_Controleur.lier(G_Controleur.couches_graph[couche_adapt.couche_id].couche_id,G_Controleur.couches_graph[couche_id_2].couche_id)
                return 
                
        if verification_taille==True:
            print("Lien entre %d et %d"%(couche_id_1,couche_id_2))
            G_Controleur.graph.edge(str(G_Controleur.couches_graph[couche_id_1].couche_id),str(G_Controleur.couches_graph[couche_id_2].couche_id))
            G_Controleur.couches_graph[couche_id_2].parent.append(G_Controleur.couches_graph[couche_id_1].couche_id)
            G_Controleur.couches_graph[couche_id_2].parents.append(G_Controleur.couches_graph[couche_id_1].couche_id)
            G_Controleur.couches_graph[couche_id_2].parents = list(dict.fromkeys(G_Controleur.couches_graph[couche_id_2].parents))
            for p in G_Controleur.couches_graph[couche_id_2].parents:
                if p not in G_Controleur.couches_graph[couche_id_2].parents:
                    G_Controleur.couches_graph[couche_id_2].parents.append(p)
            G_Controleur.couches_graph[couche_id_1].enfant.append(G_Controleur.couches_graph[couche_id_2].couche_id)

