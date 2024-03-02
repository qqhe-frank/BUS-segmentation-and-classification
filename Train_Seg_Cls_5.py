import math
import os
import random
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.autograd import Variable
import all_transfroms
from torchvision import transforms
from datasets import ImageFolder
from torch.utils.data import DataLoader,SubsetRandomSampler
from model_mt import UNet
from Model_Our import BaseLine, BaseLine_1
# from SModel.MTLNet import BaseLine
from SModel.networks.multi_task_unet import MT_Net
# from Model_Last import BaseLine38
import torch.nn as nn
from utils import DiceLoss, dice_coef, AutomaticWeightedLoss, \
    AutoWeightedLoss, AutoLoss, metric_seg, cmp_3, metric_seg_1
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


train_path = "./Dataset/"

joint_transforms=all_transfroms.Compose([
    all_transfroms.Resize((256,256)),
    all_transfroms.RandomHorizontallyFlip(0.6),
    all_transfroms.RandomRotate(30),
    all_transfroms.RandomVerticalFlip(0.6)
    ])

val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.330, 0.330, 0.330], [0.204, 0.204, 0.204])
])

# ,
#     transforms.Normalize([0.248, 0.248, 0.248], [0.151, 0.151, 0.151])

transform = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize([0.330, 0.330, 0.330], [0.204, 0.204, 0.204])
])

# ,
#     transforms.Normalize([0.248, 0.248, 0.248], [0.151, 0.151, 0.151])

#transforms.Normalize([0.330, 0.330, 0.330], [0.204, 0.204, 0.204])

target_transform = transforms.ToTensor()
val_target_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])  #transforms.Resize((256,256)),



def main():

    train_set = ImageFolder(train_path, joint_transforms, transform, target_transform)
    test_set = ImageFolder(train_path, None, val_transform, val_target_transform)

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1

    num_epochs = 90

    tr_loss = []
    val_loss = []
    test_hd95 = []
    test_asd = []
    test_ji = []
    test_dice = []
    test_acc = []
    test_pre = []
    test_recall = []
    test_f1 = []

    for train_idx,test_idx in cv.split(train_set):

        print("\nCross validation fold %d" % fold)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_set, batch_size=8,
                              num_workers=1, shuffle=False, sampler=train_sampler)

        val_loader = DataLoader(test_set, batch_size=2,
                                  num_workers=1, shuffle=False, sampler=test_sampler)

        test_loader = DataLoader(test_set, batch_size=1, num_workers=1,
                             shuffle=False, sampler=test_sampler)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = base2()
        net = net.to(device)

        train_loss, avg_val_loss, pre_ji, pre_dice, \
            hd95, asd, pre_class_acc, pre, recall, f1 = train(net, train_loader, val_loader, test_loader, fold, num_epochs)

        tr_loss.append(train_loss)
        val_loss.append(avg_val_loss)
        test_ji.append(pre_ji)
        test_dice.append(pre_dice)
        test_hd95.append(hd95)
        test_asd.append(asd)
        test_acc.append(pre_class_acc)
        test_pre.append(pre)
        test_recall.append(recall)
        test_f1.append(f1)

        fold += 1
        torch.cuda.empty_cache()

    print('\n', '#' * 10, '最终5折交叉验证结果', '#' * 10)
    print('Average Train Loss:{:.4f}'.format(np.mean(tr_loss)))
    print('Average Val Loss:{:.4f}'.format(np.mean(val_loss)))
    print('\n', '#' * 10, 'Segmentation Results', '#' * 10)
    print('Average Test Jaccard:{:.2%}±{:.4} '.format(np.mean(test_ji), np.std(test_ji)))
    print('Average Test Dice:{:.2%}±{:.4}'.format(np.mean(test_dice), np.std(test_dice)))
    print('Average Test HD95:{:.2f}±{:.4}'.format(np.mean(test_hd95), np.std(test_hd95)))
    print('Average Test ASD:{:.2f}±{:.4}'.format(np.mean(test_asd), np.std(test_asd)))
    print('\n', '#' * 10, 'Classification Results', '#' * 10)
    print('Average Test Accuracy:{:.2%}±{:.4}'.format(np.mean(test_acc), np.std(test_acc)))
    print('Average Test Precision:{:.2%}±{:.4}'.format(np.mean(test_pre), np.std(test_pre)))
    print('Average Test Recall:{:.2%}±{:.4}'.format(np.mean(test_recall), np.std(test_recall)))
    print('Average Test F1 Score:{:.2%}±{:.4}'.format(np.mean(test_f1), np.std(test_f1)))



class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, val_loader, test_loader, fold, num_epochs):

    global train_loss, class_acc, avg_val_loss, seg_dice
    avg_train_loss = AvgMeter()
    avg_train_dice = AvgMeter()
    avg_train_ji = AvgMeter()

    bce_logit = DiceLoss().cuda()
    #bc_class = nn.BCEWithLogitsLoss().cuda()
    bc_class = nn.CrossEntropyLoss().cuda()
    #awl = AutomaticWeightedLoss(2).cuda()
    awl = AutomaticWeightedLoss().cuda()

    train_loss_pic = []
    val_loss_pic = []
    val_dice_pic = []
    val_ji_pic = []
    train_ji_pic = []
    train_dice_pic = []

    params = [p for p in model.parameters() if p.requires_grad]
    params.append(awl.params)

    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.1) + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    all_result = 0.0

    save_path = './Result/pth/Train_{}_folder_model.pth'.format(fold)

    for epoch in range(num_epochs):
        model.train()

        for step, data in enumerate(train_loader):
            optimizer.zero_grad()

            images, masks, labels = data

            images = Variable(images).cuda()
            masks = Variable(masks).cuda()
            labels = Variable(labels).cuda()

            class_logits, seg_logits = model(images)

            seg_loss = bce_logit(seg_logits, masks)
            class_loss = bc_class(class_logits, labels)
            #seg_loss.backward(retain_graph=True)
            loss = awl(seg_loss, class_loss)
            #loss = 0.7*seg_loss+0.3*class_loss
            dc, jc = metric_seg_1(seg_logits, masks)

            loss.backward()
            optimizer.step()

            avg_train_loss.update(loss.item(), images.size(0))
            avg_train_dice.update(dc, images.size(0))
            avg_train_ji.update(jc, images.size(0))


        train_loss = avg_train_loss.avg
        train_dc = avg_train_dice.avg
        train_jc = avg_train_ji.avg

        train_loss_pic.append(train_loss)
        train_dice_pic.append(train_dc)
        train_ji_pic.append(train_jc)

        avg_val_loss, seg_dice, ji, hd95, \
        asd, class_acc, pre, recall, f1score = validate(model, val_loader)

        scheduler.step()
        val_loss_pic.append(avg_val_loss)
        val_dice_pic.append(seg_dice)
        val_ji_pic.append(ji)

        all_re = 0.7*seg_dice + 0.3*class_acc
        if all_re > all_result:
            all_result = all_re
            torch.save(model.state_dict(), save_path)

        print('Epoch:{}| TrainLoss:{:.4f} ValidLoss:{:.4f}| '
              'Acc:{:.2%} Pre:{:.2%} Recall:{:.2%} F1:{:.2%}|'
              'Dice:{:.2%} JA:{:.2%} HD95:{:.2f} ASD:{:.2f}.'
              .format(epoch + 1, train_loss, avg_val_loss, class_acc,
                      pre, recall, f1score, seg_dice,ji, hd95, asd))

    pre_ji, pre_dice, hd95, asd, pre_class_acc, pre, recall, f1 = test(save_path, model, test_loader, fold)


#np.around(np.mean(train_ji_pic), 3)

#=======================================================================
    epochsn = np.arange(1, len(train_loss_pic) + 1, 1)
    plt.figure(figsize=(18, 5))
    # figsize:指定figure的宽和高，单位为英寸；
    plt.subplot(131)
    y_sm = gaussian_filter1d(val_loss_pic, sigma=1)
    # 一个figure对象包含了多个子图，可以使用subplot（）函数来绘制子图：
    plt.plot(epochsn, train_loss_pic, 'b', label='Training Loss')
    plt.plot(epochsn, y_sm, 'r', label='Validation Loss')

    plt.grid(color='gray', linestyle='--')
    plt.legend()
    # plt.legend（）函数主要的作用就是给图加上图例
    plt.title('Loss, Epochs={}, Batch={}'.format(num_epochs, 5))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(132)
    plt.plot(epochsn, train_dice_pic, 'g', label='Train Dice')
    y2_sm = gaussian_filter1d(val_dice_pic, sigma=1)
    plt.plot(epochsn, y2_sm, 'cyan', label='Validation Dice')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Dice coefficient score')
    plt.xlabel('Epochs')
    plt.ylabel('CSC')

    plt.subplot(133)
    plt.plot(epochsn, train_ji_pic, 'chocolate', label='Train Jaccard')
    y3_sm = gaussian_filter1d(val_ji_pic, sigma=1)
    plt.plot(epochsn, y3_sm, 'm', label='Validation Jaccard')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Jaccard coefficient score')
    plt.xlabel('Epochs')
    plt.ylabel('CSC')
    plt.savefig('./Result/pic/savefig_{}.png'.format(fold))
    plt.show()


    return train_loss, avg_val_loss, pre_ji, pre_dice, hd95, asd, pre_class_acc, pre, recall, f1


def validate(model, val_loader):
    losses = AvgMeter()
    avg_val_dice = AvgMeter()
    avg_val_jc = AvgMeter()
    avg_val_hd = AvgMeter()
    avg_val_asd = AvgMeter()


    bce_logit = DiceLoss().cuda()
    #class_logit = nn.BCEWithLogitsLoss().cuda()
    class_logit = nn.CrossEntropyLoss().cuda()
    #awl = AutomaticWeightedLoss(2).cuda()
    awl = AutomaticWeightedLoss().cuda()

    val_preds = []
    val_trues = []

    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
        for i, (input, target, label) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            label = label.cuda()

            pre_class, output = model(input)

            seg_loss = bce_logit(output, target)
            class_loss = class_logit(pre_class, label)
            loss = awl(seg_loss, class_loss)
            #loss = 0.6*seg_loss+0.4*class_loss
            dc, jc, hdc, asdc = metric_seg(output, target)

            avg_val_dice.update(dc, input.size(0))
            avg_val_jc.update(jc, input.size(0))
            avg_val_hd.update(hdc, input.size(0))
            avg_val_asd.update(asdc, input.size(0))

            pre_class = torch.sigmoid(pre_class)
            predict_class = torch.max(pre_class, dim=1)[1]
            val_preds.extend(predict_class.detach().cpu().numpy())
            val_trues.extend(label.detach().cpu().numpy())

            losses.update(loss.item(), input.size(0))
    val_loss = losses.avg

    sklearn_accuracy = accuracy_score(val_trues, val_preds)
    sklearn_precision = precision_score(val_trues, val_preds, average='weighted')
    sklearn_recall = recall_score(val_trues, val_preds, average='macro')
    sklearn_f1 = f1_score(val_trues, val_preds, average='weighted')


    return val_loss, np.around(avg_val_dice.avg, 3), np.around(avg_val_jc.avg, 3), \
        np.around(avg_val_hd.avg, 3), np.around(avg_val_asd.avg, 3),\
        sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1


def test(save_path, model, test_loader, fold):

    JI=[]
    Dices=[]
    test_preds = []
    test_trues = []
    HD95_1 = []
    ASD_1 = []
    weights_path = save_path
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path))
    to_pil = transforms.ToPILImage()

    model.eval()
    with torch.no_grad():
      for i, (input, target, label) in enumerate(test_loader):

          image = Variable(input).cuda()
          target = Variable(target).cuda()
          label = Variable(label).cuda()

          pro_class, pro_seg = model(image)
          pro_class = torch.sigmoid(pro_class)
          predict_class = torch.max(pro_class, dim=1)[1]
          test_preds.extend(predict_class.detach().cpu().numpy())
          test_trues.extend(label.detach().cpu().numpy())

          a = target.squeeze(0)
          b = image.squeeze(0)
          a = to_pil(a)
          b = to_pil(b)
          a.save('./Result/TestResult/{}/mask{}.png'.format(fold, i))
          b.save('./Result/TestResult/{}/img{}.png'.format(fold, i))

          pro=torch.sigmoid(pro_seg).data.squeeze(0).cpu()
          c = to_pil(pro)
          c.save('./Result/TestResult/{}/pre{}.png'.format(fold, i))
          target = target.squeeze(0).cpu()
          pro = np.array(pro)
          target = np.array(target)
          pro[pro>=0.5]=1
          pro[pro<0.5]=0
          TP=float(np.sum(np.logical_and(pro==1,target==1)))
          TN=float(np.sum(np.logical_and(pro==0,target==0)))
          FP=float(np.sum(np.logical_and(pro==1,target==0)))
          FN=float(np.sum(np.logical_and(pro==0,target==1)))
          JA=TP/((TP+FN+FP)+1e-5)
          DI=2*TP/((2*TP+FN+FP+1e-5))

          Dices.append(DI)
          JI.append(JA)
          hd95, asd = cmp_3(pro, target)
          HD95_1.append(hd95)
          ASD_1.append(asd)

    sklearn_accuracy = accuracy_score(test_trues, test_preds)
    sklearn_precision = precision_score(test_trues, test_preds, average='weighted')
    sklearn_recall = recall_score(test_trues, test_preds, average='macro')
    sklearn_f1 = f1_score(test_trues, test_preds, average='weighted')

    print('Test Result:\n''Segmentation:\n Jaccard:{:.2%} '
          'Dice:{:.2%} HD95:{:.2f} ASD:{:.2f}'.format(np.around(np.mean(JI), 3),
                                                      np.around(np.mean(Dices), 3),
                                                      np.around(np.mean(HD95_1), 3),
                                                      np.around(np.mean(ASD_1), 3)))
    print('Classification:\n Accuary:{:.2%} '
          'Precision:{:.2%} Recall:{:.2%} Score:{:.2%}'.format(sklearn_accuracy,
                                                      sklearn_precision,
                                                      sklearn_recall,
                                                      sklearn_f1))


    return np.around(np.mean(JI), 3), np.around(np.mean(Dices), 3), np.around(np.mean(HD95_1), 3),\
        np.around(np.mean(ASD_1), 3),sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1





if __name__ == "__main__":
    seed = 2
    #seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    main()








