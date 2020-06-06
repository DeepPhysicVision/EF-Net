import h5py
import numpy as np
from FileReader import FileReader
from Evaluator import Evaluator
import AttentionCapsule as models
import cv2
import skimage.io as io
from keras.preprocessing import image
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

if __name__ == '__main__':
    model_path = 'Result/'
    train_image_paths = './Data/2015Image/TrainImage/'
    test_image_paths = './Data/2015Image/TestImage/'
    val_image_paths = './Data/2015Image/ValImage/'
    file_reader = FileReader()

    # train input=图片
    train_images = glob.glob(train_image_paths + '*.jpg')
    train_image = []
    for i in range(len(train_images)):
        img = image.load_img(train_images[i], target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.astype('float32')
        img = img/255
        img = np.expand_dims(img, axis=0)  # 增加第一个batch维度
        train_image.append(img)
    train_image = np.concatenate([x for x in train_image])  # 把所有图片数组concatenate在一起，便于批量处理

    # test input=图片
    test_images = glob.glob(test_image_paths + '*.jpg')
    test_image = []
    for i in range(len(test_images)):
        img = image.load_img(test_images[i], target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.astype('float32')
        img = img/255
        img = np.expand_dims(img, axis=0)  # 增加第一个batch维度
        test_image.append(img)
    test_image = np.concatenate([x for x in test_image])  # 把所有图片数组concatenate在一起，便于批量处理

    # val input=图片
    val_images = glob.glob(val_image_paths + '*.jpg')
    val_image = []
    for i in range(len(val_images)):
        img = image.load_img(val_images[i], target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.astype('float32')
        img = img / 255
        img = np.expand_dims(img, axis=0)  # 增加第一个batch维度
        val_image.append(img)
    val_image = np.concatenate([x for x in val_image])  # 把所有图片数组concatenate在一起，便于批量处理

    # train/test/val input=句子+位置, out=标签
    train_aspect_labels, train_aspect_text_inputs, train_sentence_inputs, _ = file_reader.load_inputs_and_label(name='train')
    train_sentence_inputs, train_aspect_text_inputs, train_positions, _ = file_reader.get_position_input(train_sentence_inputs,train_aspect_text_inputs)

    test_aspect_labels, test_aspect_text_inputs, test_sentence_inputs, test_true_labels = file_reader.load_inputs_and_label(name='test')
    test_sentence_inputs, test_aspect_text_inputs, test_positions, _ = file_reader.get_position_input(test_sentence_inputs,test_aspect_text_inputs)

    val_aspect_labels, val_aspect_text_inputs, val_sentence_inputs, val_true_labels = file_reader.load_inputs_and_label(name='val')
    val_sentence_inputs, val_aspect_text_inputs, val_positions, _ = file_reader.get_position_input(val_sentence_inputs, val_aspect_text_inputs)

    # 额外输入 position embedding矩阵
    position_matrix = file_reader.load_position_matrix()
    embedding_matrix = file_reader.get_embedding_matrix()

    position_ids = file_reader.get_position_ids(max_len=36)
    file_reader.convert_position(position_inputs=train_positions, position_ids=position_ids)
    file_reader.convert_position(position_inputs=test_positions, position_ids=position_ids)

    # train/test/val input=目标
    train_aspects = file_reader.pad_aspect_index(train_aspect_text_inputs.tolist(), max_length=8)
    test_aspects = file_reader.pad_aspect_index(train_aspect_text_inputs.tolist(), max_length=8)
    val_aspects = file_reader.pad_aspect_index(val_aspect_text_inputs.tolist(), max_length=8)

    # 载入模型
    model = models.build_model(max_len=36,aspect_max_len=8,embedding_matrix=embedding_matrix,position_embedding_matrix=position_matrix)
    evaluator = Evaluator(true_labels=test_true_labels, sentences=test_sentence_inputs, aspects=test_aspect_text_inputs)

    epoch = 1
    while epoch <= 100:
        model.fit([train_image, train_sentence_inputs, train_positions,train_aspects], train_aspect_labels,
              validation_data=([val_image,val_sentence_inputs, val_positions,val_aspects], val_aspect_labels),
                  epochs=1,batch_size=128, verbose=2)

        results = model.predict([test_image,test_sentence_inputs,test_positions,test_aspects],batch_size=128, verbose=2)
        print("\n-epoch"+str(epoch)+"-")
        F, acc = evaluator.get_macro_f1(predictions=results, epoch=epoch)
        if epoch % 5 == 0:
            print("max f1"+str(evaluator.max_F1))
            print("max f1 is gained in epoch"+str(evaluator.max_F1_epoch))
            print("max acc"+str(evaluator.max_acc))
            print("max acc is gained in epoch"+str(evaluator.max_acc_epoch))

        if acc > 0.7250:
            model.save(model_path+"Acc_"+str(acc*100)+"+F_"+str(F*100)+"+Epoch_"+str(epoch)+".h5")
        elif F > 0.6700:
            model.save(model_path + "Acc" + str(acc * 100) + "F" + str(F * 100) + "Epoch" + str(epoch)+".h5")
        epoch += 1


