from keras import backend as K
K.set_learning_phase(0) # Empêche l'entrainement

class DeepDream:
    def __init__(self,model,image_path):
        """ Visualise suivant la méthode du réseau de neurone Google DeepDream ce que détecte chaque couche
        - <i>model</i>: <b>Modèle keras</b>, modèle à évaluer
        - <i>image_path</i>: <b>string</b>, chemin vers l'image qui servira de référence pour détecter le rôle de chaque couche
        
        Les images générées par chaque couche seront disposées dans le dossier ./analyse/deepdream/nom_couche/....
        """
        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])
    def compute_(self,model):
        for layer_name in self.layer_dict.keys():
            # Construction du coût
            loss = K.variable(0.) 
            activation = self.layer_dict[layer_name].output
            scaling = K.prod(K.cast(K.shape(activation), 'float32'))
            ## Le slicing est expliqué dans le livre : You avoid border artifacts by only involving nonborder pixels in the loss
            loss += K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
            # Remontée
            dream = model.input
            grads = K.gradients(loss, dream)[0]
            grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) # Normalizes the gradients (important trick)
            outputs = [loss, grads]
            fetch_loss_and_grads = K.function([dream], outputs) # Sets up a Keras function to retrieve the value of the loss and gradients, given an input image
            def eval_loss_and_grads(x):
                outs = fetch_loss_and_grads([x])
                loss_value = outs[0]
                grad_values = outs[1]
                return loss_value, grad_values
            def gradient_ascent(x, iterations, step, max_loss=None):
                for i in range(iterations):
                    loss_value, grad_values = eval_loss_and_grads(x)
                    if max_loss is not None and loss_value > max_loss:
                        break
                    print('...Loss value at', i, ':', loss_value)
                    x += step * grad_values
            return x
