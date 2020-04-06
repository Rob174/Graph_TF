import numpy as np
np.random.seed(5)
#Graines exemples intéressants : 
"""
Avec 3 Conv (et 1 Input)
1 : pour ééviter les retours en arrière
3 : pour éviter les liens en doublon
5 : pour vérifier que l'input est bien fournie à tous les layers sans parent
Avec Input Conv Pool Conv

"""
class HP_test:
    def __init__(self):
        pass
    def Choice(self,name,possibilites,default=None):
        choix=np.random.randint(0,len(possibilites))
        return possibilites[choix]
    def Int(self,name,min_value,max_value,default=None):
        return max_value
