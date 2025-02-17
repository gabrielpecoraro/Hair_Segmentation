
import torch

def cross_entropy_loss(y_pred, y_true):
    """
    y_pred: tensor of shape (batch_size, 2, h, w)
    y_true: tensor of shape (batch_size, h, w)
    """
    # Pour éviter des valeurs négatives ou supérieures à 1
    y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7) 
    # Convertir les étiquettes en entiers pour indexer les prédictions
    y_true = y_true.long()
    # Initialisation de la loss
    loss = 0
    # Pour chaque élément du batch
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[2]):
            for k in range(y_pred.shape[3]):
                # Récupération de la probabilité prédite pour la classe vraie
                pred_for_true_class = y_pred[i, 0,j,k]
                # Calcul de la binary cross-entropy pour l'élément i
                loss_i = -y_true[i,j,k]*torch.log(pred_for_true_class)-(1-y_true[i,j,k])*torch.log(1-pred_for_true_class)
                # Ajout de la loss pour l'élément i à la loss globale
                loss += loss_i
    # Calcule de la loss moyenne pour le batch
    loss = loss/(y_pred.shape[0]*y_pred.shape[2]*y_pred.shape[3])
    return loss
