import tensorflow as tf
import numpy as np


def model(X):
    X = X / 255

    conv1 = tf.layers.batch_normalization(tf.layers.conv2d(X, 64, 6, activation=tf.nn.leaky_relu, padding="SAME"))
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.batch_normalization(tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.leaky_relu, padding="SAME"))
    conv3 = tf.layers.batch_normalization(tf.layers.conv2d(conv2, 128, 1, activation=tf.nn.leaky_relu, padding="SAME"))
    pool2 = tf.layers.max_pooling2d(conv3, 2, 2)

    conv4 = tf.layers.batch_normalization(tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.leaky_relu, padding="SAME"))
    conv5 = tf.layers.batch_normalization(tf.layers.conv2d(conv4, 256, 1, activation=tf.nn.leaky_relu, padding="SAME"))
    conv6 = tf.layers.batch_normalization(tf.layers.conv2d(conv5, 256, 3, activation=tf.nn.leaky_relu, padding="SAME"))
    pool3 = tf.layers.max_pooling2d(conv6, 2, 2)

    conv7 = tf.layers.batch_normalization(tf.layers.conv2d(pool3, 512, 1, activation=tf.nn.leaky_relu, padding="SAME"))
    conv8 = tf.layers.batch_normalization(tf.layers.conv2d(conv7, 512, 3, activation=tf.nn.leaky_relu, padding="SAME"))
    conv9 = tf.layers.batch_normalization(tf.layers.conv2d(conv8, 512, 1, activation=tf.nn.leaky_relu, padding="SAME"))
    pool4 = tf.layers.max_pooling2d(conv9, 2, 2)

    conv10 = tf.layers.batch_normalization(tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.leaky_relu, padding="SAME"))

    out = tf.layers.conv2d(conv10, 25, 1)
    return out

def calc_iou(boxes1,boxes2):
    #convert [x,y,w,h] to [x1,y1,x2,y2]

    boxes1_t=tf.stack([
        boxes1[..., 0] - boxes1[...,2] / 2.0,
        boxes1[..., 1] - boxes1[...,3] / 2.0,
        boxes1[..., 0] + boxes1[...,2] / 2.0,
        boxes1[..., 1] + boxes1[...,3] / 2.0],
    axis=-1)
    boxes2_t = tf.stack([
        boxes2[..., 0] - boxes2[..., 2] / 2.0,
        boxes2[..., 1] - boxes2[..., 3] / 2.0,
        boxes2[..., 0] + boxes2[..., 2] / 2.0,
        boxes2[..., 1] + boxes2[..., 3] / 2.0],
    axis=-1)

    #calc left up and right down
    lu = tf.maximum(boxes1_t[...,:2],boxes2_t[...,:2])
    rd = tf.minimum(boxes1_t[...,2:],boxes2_t[...,2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    #calculate boxes1 and boxes2 square
    square1 = boxes1[...,2] * boxes1[...,3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss(pred, out):
    out_score = out[..., 4]
    pred_score = pred[..., 4]
    pred_cat = pred[..., 5:]
    out_cat = out[..., 5:]
    pred_boxes = pred[...,0:4]
    out_boxes = out[...,0:4]

    obj=out_score
    no_obj = tf.ones_like(out_score) - out_score

    pred_boxes=tf.stack([
        pred_boxes[...,0],
        pred_boxes[...,1],
        tf.square(pred_boxes[...,2]),
        tf.square(pred_boxes[...,3])],
    axis=-1)

    iou=calc_iou(pred_boxes,out_boxes)

    #object loss
    loss_obj= obj * tf.square(iou - pred_score)
    loss_obj = tf.reduce_mean(tf.reduce_sum( loss_obj,axis=[1,2] ))

    #no object loss
    loss_noobj = no_obj * tf.square(pred_score)
    loss_noobj = tf.reduce_mean(tf.reduce_sum(loss_noobj, axis=[1, 2]))

    # x,y,w,h loss
    pred_boxes_l=tf.stack([
        pred_boxes[...,0],
        pred_boxes[...,1],
        pred_boxes[...,2],
        pred_boxes[...,3]],
    axis=-1)

    loss_box = tf.reduce_sum( tf.square(pred_boxes_l - out_boxes), axis=-1)
    loss_box = obj * loss_box
    loss_box = tf.reduce_mean( tf.reduce_sum (loss_box,axis=[1,2] ) )

    #class loss
    loss_class = tf.reduce_sum( tf.square(out_cat - pred_cat),axis=-1 )
    loss_class = obj * loss_class
    loss_class = tf.reduce_mean( tf.reduce_sum(loss_class ,axis=[1,2]))

    return loss_class + loss_obj + .5 * loss_noobj + 5 * loss_box


