import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import *

def display(display_list,num=1):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if display_list[i].ndim == 3:
    	plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    else:
    	plt.imshow(display_list[i])
    plt.axis('off')
  plt.savefig("./results/output_{}.png".format(num))

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    idx=0
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)],num)
      idx+=1
    # else:
      # display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))])



if __name__ == "__main__":
	x_train,x_test,y_train,y_test = load_data()
	train_ds,test_ds = convert_tf_dataset(x_train,x_test,y_train,y_test,BATCH_SIZE=32)

	#load saved model
	model = tf.keras.models.load_model("./model/unet")
	# model.summary()
	show_predictions(test_ds,3)


