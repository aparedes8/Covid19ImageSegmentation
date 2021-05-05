
import numpy as np
import tensorflow as tf
from layers import *
from load import *
from vis import display,create_mask
from dataaug import genDataAugImages
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize

####################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

####################################

(x_train, y_train), (x_test, y_test) = load()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

print(x_train.shape)
# x_train, y_train = genDataAugImages(x_train, y_train)

####################################

tmp, counts = np.unique(y_train, return_counts=True)
assert (np.all(tmp == [0,1,2,3]))

'''
weight = counts / np.sum(counts)
'''

'''
alpha_0 = counts[0] / np.sum(counts)
alpha_1 = counts[1] / np.sum(counts)
alpha_2 = counts[2] / np.sum(counts)
alpha_3 = counts[3] / np.sum(counts)
'''

# A value pos_weight > 1 decreases the false negative count, hence increasing the recall. 
# Conversely setting pos_weight < 1 decreases the false positive count and increases the precision. 
# This can be seen from the fact that pos_weight is introduced as a multiplicative coefficient for the positive labels term in the loss expression:

weight = counts / np.sum(counts)
weight = 1. / weight
weight = weight / np.max(weight)
weight_tf = tf.constant(weight, dtype=tf.float32)
print(weight_tf)
weight_tf = weight_tf * 1000

####################################

model = model(layers=[

conv_block((3,3,1,32)),
max_pool(2),

conv_block((3,3,32,32)),
max_pool(2),

conv_block((3,3,32,64)),
max_pool(2),

conv_block((3,3,64,64)),
up_pool(2),

conv_block((3,3,64,32)),
up_pool(2),

conv_block((3,3,32,32)),
up_pool(2),

final_conv_block((3,3,32,4))
])

####################################

params = model.get_params()
optimizer = tf.keras.optimizers.Adam(lr=0.01)

####################################

def pred(model, x):
    return model.train(x)

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=-1)
        ################################################################################################################
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
        ################################################################################################################
        y_one_hot = tf.one_hot(y, depth=4)
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_one_hot, logits=pred_logits, pos_weight=weight_tf)
        ################################################################################################################
        ################################################################################################################
    grad = tape.gradient(loss, params)
    return loss, grad, pred_label

####################################

def eval(labels, preds):
    assert (np.shape(labels) == np.shape(preds))
    assert (len(np.shape(labels)) == 4)
    correct = (labels == preds) * labels
    correct = np.sum(correct, axis=(0,1,2))
    total = np.sum(labels, axis=(0,1,2))
    assert (np.all(correct <= total))
    return correct, total

####################################
#training
prec = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
train_acc = []
train_loss = []
test_acc = []
test_loss = []
epoch_list = []
batch_size = 10
epochs = 30
print("\n\n\n")
for i in range(epochs):
    _correct, _total,_loss = 0, 0,0
    print("EPOCH {}".format(i+1))
    for batch in range(0, len(x_train), batch_size):
        xs = x_train[batch:batch+batch_size]
        ys = y_train[batch:batch+batch_size]
        
        loss, grad, pred = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, params))

        label_np = tf.one_hot(ys, depth=4, axis=-1).numpy()
        pred_np = tf.one_hot(pred, depth=4, axis=-1).numpy()
        correct, total = eval(labels=label_np, preds=pred_np)

        _correct += correct
        _total += total
        _loss += loss
        
    accuracy = _correct / _total #* 100
    dice = (2 *_correct)/((2 *_correct)+(_total-_correct))
    #precision = 
    mean_dice = np.mean(dice)
    mean_loss = np.mean(_loss)
    train_acc.append(mean_dice)
    train_loss.append(mean_loss)
    epoch_list.append(i+1)

    print("Categorical accuracy during training {}".format(accuracy))
    print("DICE accuracy during training {}".format(dice))

    ####################################
    _correct, _total,_loss = 0, 0, 0
    loss, grad, pred = gradients(model, x_test[0:10], y_test[0:10])

    label_np = tf.one_hot(ys, depth=4, axis=-1).numpy()
    pred_np = tf.one_hot(pred, depth=4, axis=-1).numpy()
    correct, total = eval(labels=label_np, preds=pred_np)
    prec.update_state(label_np,pred_np)
    print(prec.result().numpy())

    _correct += correct
    _total += total
    _loss += loss
    # print(_loss)
        
    accuracy = _correct / _total #* 100
    dice = (2 *_correct)/((2 *_correct)+(_total-_correct))
    mean_dice = np.mean(dice)
    mean_loss = np.mean(_loss)
    test_acc.append(mean_dice)
    test_loss.append(mean_loss)
    print("\n\n\n")
    print("Categorical accuracy during testing {}".format(accuracy))

####################################
#show results of segmeentation
for i in range(10):
    img = x_test[i]
    pred_logits= model.train(img[np.newaxis,...])  
    pred_label = create_mask(pred_logits)
    # plt.figure()
    display([img,y_test[i],pred_label],i)
####################################
#plot accuracys and precisions


plt.figure()
plt.title("DICE Accuracy vs Epoch Plot")
plt.plot(epoch_list,train_acc,label='Train Accuracy',color='r')
plt.plot(epoch_list,test_acc,label='Test Accuracy',color='g')

plt.xlabel("EPOCHS")
plt.ylabel("DICE Accuracy")
plt.legend()
plt.savefig("./results/Accuracy_plot.png")

plt.figure()
plt.title("Loss vs Epoch Plot")
plt.plot(epoch_list,train_loss,label='Train loss',color='r')
plt.plot(epoch_list,test_loss,label='Test loss',color='g')
plt.xlabel("EPOCHS")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./results/loss_plot.png")




# def plot_roc(name, labels, predictions, **kwargs):
#   fp, tp, _ = roc_curve(labels, predictions)

#   plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
#   plt.xlabel('False positives [%]')
#   plt.ylabel('True positives [%]')
#   plt.xlim([-0.5,20])
#   plt.ylim([80,100.5])
#   plt.grid(True)
#   ax = plt.gca()
#   ax.set_aspect('equal')

# plt.figure()

# # pred_logits= model.train(x_test[...])
# # pred_label = create_mask(pred_logits)

# print(y_test.shape)
# print(pred.shape)
# y = y_test[0:10].flatten()
# # Binarize the output
# y = label_binarize(y, classes=[0, 1, 2, 3])

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(4):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# pred = pred.numpy().flatten()
# print(y.shape)
# print(pred.shape)
# plot_roc("Test Baseline",y,pred)
# plt.savefig("./results/roc_plot.png")