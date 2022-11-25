import torch
from torch.nn import functional as F


beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-10
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self):
        print ("semantic loss functions initialized")

    # def dice_coef(self, y_true, y_pred):
    #     y_true_f = K.flatten(y_true)
    #     y_pred_f = K.flatten(y_pred)
    #     intersection = K.sum(y_true_f * y_pred_f)
    #     return (2. * intersection + K.epsilon()) / (
    #                 K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    # def sensitivity(self, y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     return true_positives / (possible_positives + K.epsilon())

    # def specificity(self, y_true, y_pred):
    #     true_negatives = K.sum(
    #         K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    #     possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    #     return true_negatives / (possible_negatives + K.epsilon())

    # def convert_to_logits(self, y_pred):
    #     y_pred = torch.clip(y_pred, epsilon, 1-epsilon)
    #     # y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
    #     #                           1 - tf.keras.backend.epsilon())
    #     return tf.math.log(y_pred / (1 - y_pred))

    # def weighted_cross_entropyloss(self, y_true, y_pred):
    #     def weighted_cross_entropy_with_logits(logits, target, pos_weight):
    #         return target * -logits.sigmoid().log() * pos_weight + \
    #                 (1 - target) * -(1 - logits.sigmoid()).log()
    #     y_pred = self.convert_to_logits(y_pred)
    #     pos_weight = beta / (1 - beta)
    #     loss = weighted_cross_entropy_with_logits(y_pred, y_true, pos_weight)
    #     return torch.mean(loss)
    #     # loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
    #     #                                                 targets=y_true,
    #     #                                                 pos_weight=pos_weight)
    #     # return tf.reduce_mean(loss)

    # def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
    #     weight_a = alpha * (1 - y_pred) ** gamma * targets
    #     weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    #     return (torch.log1p(torch.exp(-torch.abs(logits))) + torch.relu(-logits)) * \
    #         (weight_a + weight_b) + logits * weight_b
    #     # return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
    #     #     -logits)) * (weight_a + weight_b) + logits * weight_b

    # def focal_loss(self, y_true, y_pred):
    #     y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
    #                               1 - tf.keras.backend.epsilon())
    #     logits = tf.math.log(y_pred / (1 - y_pred))

    #     loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
    #                                   alpha=alpha, gamma=gamma, y_pred=y_pred)

    #     return tf.reduce_mean(loss)

    # def depth_softmax(self, matrix):
    #     sigmoid = lambda x: 1 / (1 + K.exp(-x))
    #     sigmoided_matrix = sigmoid(matrix)
    #     softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    #     return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = F.binary_cross_entropy_with_logits(y_true, y_pred) + \
               self.dice_loss(torch.sigmoid(y_true), torch.sigmoid(y_pred))
        return loss / 2.0

    # def confusion(self, y_true, y_pred):
    #     smooth = 1
    #     y_pred_pos = K.clip(y_pred, 0, 1)
    #     y_pred_neg = 1 - y_pred_pos
    #     y_pos = K.clip(y_true, 0, 1)
    #     y_neg = 1 - y_pos
    #     tp = K.sum(y_pos * y_pred_pos)
    #     fp = K.sum(y_neg * y_pred_pos)
    #     fn = K.sum(y_pos * y_pred_neg)
    #     prec = (tp + smooth) / (tp + fp + smooth)
    #     recall = (tp + smooth) / (tp + fn + smooth)
    #     return prec, recall

    # def true_positive(self, y_true, y_pred):
    #     smooth = 1
    #     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    #     y_pos = K.round(K.clip(y_true, 0, 1))
    #     tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    #     return tp

    # def true_negative(self, y_true, y_pred):
    #     smooth = 1
    #     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    #     y_pred_neg = 1 - y_pred_pos
    #     y_pos = K.round(K.clip(y_true, 0, 1))
    #     y_neg = 1 - y_pos
    #     tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    #     return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    # def tversky_loss(self, y_true, y_pred):
    #     return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)