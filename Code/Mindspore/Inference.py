import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from mindspore import context, load_checkpoint, load_param_into_net, Model

from Nets import *
from T1.Code.Dataset import create_Dataset

'''
Setup
'''
context.set_context(device_target='CPU', mode=context.PYNATIVE_MODE)

data_flag = "T1-2-1"
data_mode = "eval"
debug = True
path_test = {
                "RSOD": "E:/ModelArts/T1/Datasets/RSOD-OD/aircraft",
                "UCMLU": "E:/ModelArts/T1/Datasets/UCMerced_LandUse",
            }
path_model = "E:/ModelArts/T1/Output/X7/F1-0.98_epoch-200.ckpt"
path_save = os.path.join("./Inference/T1-2-1/Eval","F1-0.98_epoch-200","debug" if debug else "release")
# # Super-Parameters
best_scale_factor = 10
PL_nums = [6]
F1_List = [0.314, 0.632, 0.806, 0.977, 0.974, 0.967]

'''
Post-Process
'''
def label2rgb(label):
    label2color_dict = {
        0:[0,0,0],
        1:[255,0,0],
        2:[0,255,0],
        3:[0,0,255],
        4:[255,255,0],
        5:[255,0,255],
        6:[0,255,255],
        7:[128,0,0],
        8:[0,128,0],
        9:[0,0,128],
        10:[128,128,0],
        11:[128,0,128],
        12:[0,128,128],
        13:[64,0,0],
        14:[0,64,0],
        15:[0,0,64],
        16:[64,64,0],
        17:[64,0,64],
        18:[0,64,64],
        19:[32,128,255],
        20:[128,32,255],
        21:[128,64,255],
        22:[64,128,255],
        23:[255,64,128],
    }
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno

'''
Net
'''
class Predict_Net(nn.Cell):
    def __init__(self, network, debug=False):
        super(Predict_Net, self).__init__()
        self.network = network
        self.debug = debug
        self.F1_List = F1_List
        self.best_scale_factor = best_scale_factor
        F1 = np.array(self.F1_List)
        if debug:
            self.score_weight_List = [np.exp(weight_scale_factor*F1) /\
                            np.exp(weight_scale_factor*F1).sum() for weight_scale_factor in config.scale_factor_List]
        else:
            self.score_weight_List = np.exp(self.best_scale_factor*F1) / np.exp(self.best_scale_factor*F1).sum()
        self.label2rgb = label2rgb

    def construct(self, images):
        PLset = self.network(images)
        PL_List = []
        for p in PLset:
            PL_List.append(np.squeeze((ops.Softmax(3)(p).asnumpy()).astype(np.float32)))
        # Debug Mode
        if debug:
            # un-weighted vote
            p_vote = 0
            for p in PL_List:
                p_vote += p
            p_vote = np.argmax(p_vote, axis=2)
            p_vote = self.label2rgb(p_vote)
            PL_List.append(p_vote)
            # weighted vote
            for score_weight in self.score_weight_List:
                p_vote = 0
                for PL_index in range(len(self.F1_List)):
                    p_vote += PL_List[PL_index] * score_weight[PL_index]            
                p_vote = np.argmax(p_vote, axis=2)
                p_vote = self.label2rgb(p_vote)
                PL_List.append(p_vote)
            # cal mid-result
            for PL_index in range(len(self.F1_List)):
                p_mid = PL_List[PL_index]
                p_mid = np.argmax(p_mid, axis=2)
                p_mid = self.label2rgb(p_mid)
                PL_List[PL_index] = p_mid
            return PL_List
        # Relese Mode
        else:
            p_vote = 0
            for PL_index in range(len(self.F1_List)):
                p_PL = PL_List[PL_index]
                p_vote += self.score_weight_List[PL_index] * p_PL
            p_vote = np.argmax(p_vote, axis=2)
            p_vote = self.label2rgb(p_vote)
            return p_vote

if __name__ == "__main__":
    '''
    Dataset
    '''
    dataset_test = create_Dataset(     
                                    data_flag=data_flag,                               
                                    data_path=path_test, 
                                    batch_size=1, 
                                    shuffle=True,
                                    mode=data_mode)

    '''
    Model
    '''
    # load trained net
    train_net = Train_net()
    param_dict = load_checkpoint(path_model)
    load_param_into_net(train_net, param_dict)
    # create predict net
    predict_net = Predict_Net(train_net, debug)
    model = Model(network=predict_net)

    '''
    Log
    '''
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    if debug:
        vote_columns = len(config.scale_factor_List)+1
        rows = 1+len(PL_nums)+1  # Raw, p, Vote

    '''
    Predict
    '''
    data = dataset_test.create_dict_iterator()

    for index, input in enumerate(tqdm(data)):
        output = model.predict(input["images"])
        # raw images
        if data_flag == "BCD":
            image_A = input["images"][0][0].asnumpy().transpose(1,2,0)
            image_B = input["images"][0][1].asnumpy().transpose(1,2,0)
            label = np.squeeze(input["labels"].asnumpy())
            for PL_index in range(len(F1_List)):
                output[PL_index] = np.argmax(output[PL_index], axis=2)
        elif data_flag == "GID":
            image = input["images"][0].asnumpy().transpose(1,2,0)
            image = np.argmax(image, axis=2)
            image = label2rgb(image)
            label = np.squeeze(input["labels"].asnumpy())
            label = label2rgb(label)
        elif data_flag == "RSOD-Aircraft":
            image = input["images"][0].asnumpy().transpose(1,2,0)
            label = np.squeeze(input["labels"].asnumpy())
            label = label2rgb(label)
        elif data_flag == "T1-2-1":
            # image = input["images"][0].asnumpy().transpose(1,2,0)
            image = input["images"][0].asnumpy()
            if image[0][0][0] == 0:
                # image = image[1::2,:,:].transpose(1,2,0)
                image = image[3:,:,:].transpose(1,2,0)
            else:
                image = image[:3,:,:].transpose(1,2,0)
            label = np.squeeze(input["labels"].asnumpy())
            label = label2rgb(label)
        plt.figure(dpi=100, figsize=(10,10), facecolor="#FFFFFF")
        if debug:
            # Row1-Raw
            if data_flag == "BCD":
                plt.subplot(rows,3,1)
                plt.title("A")
                plt.imshow(image_A)
                plt.subplot(rows,3,2)
                plt.title("B")
                plt.imshow(image_B)
                plt.subplot(rows,3,3)
                plt.title("Label")
                plt.imshow(label)
            elif data_flag == "GID":
                plt.subplot(rows,2,1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(rows,2,2)
                plt.title("Label")
                plt.imshow(label)
            elif data_flag == "RSOD-Aircraft":
                plt.subplot(rows,2,1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(rows,2,2)
                plt.title("Label")
                plt.imshow(label)
            elif data_flag == "T1-2-1":
                plt.subplot(rows,2,1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(rows,2,2)
                plt.title("Label")
                plt.imshow(label)
            # Row2-p
            for PL_index in range(len(PL_nums)):
                for column in range(PL_nums[PL_index]):
                    plt.subplot(rows,PL_nums[PL_index],(PL_index+1)*PL_nums[PL_index]+column+1)
                    plt.title("{}-{}".format(PL_index+1, column+1))
                    plt.xticks([]), plt.yticks([])
                    plt.imshow(output[column])
            # Row3-Vote
            for column in range(vote_columns):
                plt.subplot(rows,vote_columns, 2*vote_columns+column+1)
                if column == 0:
                    plt.title("Unweighted")
                else:
                    plt.title("SF-{}".format(config.scale_factor_List[column-1]))
                plt.imshow(output[column-vote_columns])
            # save
            plt.savefig(os.path.join(path_save, "{}.png".format(index)))
        else:
            if data_flag == "BCD":
                plt.subplot(2,2,1)
                plt.title("A")
                plt.imshow(image_A)
                plt.subplot(2,2,2)
                plt.title("B")
                plt.imshow(image_B)
                plt.subplot(2,2,3)
                plt.title("Output")
                plt.imshow(output)
                plt.subplot(2,2,4)
                plt.title("Label")
                plt.imshow(label)
            elif data_flag == "GID":
                plt.subplot(2,1,1)
                plt.title("Image")
                plt.imshow(image)
            elif data_flag == "RSOD-Aircraft":
                plt.subplot(2,1,1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(2,2,3)
                plt.title("Output")
                plt.imshow(output)
                plt.subplot(2,2,4)
                plt.title("Label")
                plt.imshow(label)
            elif data_flag == "T1-2-1":
                plt.subplot(2,1,1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(2,2,3)
                plt.title("Output")
                plt.imshow(output)
                plt.subplot(2,2,4)
                plt.title("Label")
                plt.imshow(label)

            # save
            plt.savefig(os.path.join(path_save, "{}.png".format(index)))