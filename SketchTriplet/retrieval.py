from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

def extract_feat_sketch(net, sketch_src):
    img_size = 256
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    sketch_src = transform(sketch_src)
    sketch = Variable(sketch_src.unsqueeze(0)).cuda()
    feat = net.get_branch_sketch(sketch)
    feat = feat.cpu().data.numpy()
    return feat

def get_real_path(retrieval_list):
    retrieval_list = list(retrieval_list)

    real_list_set = []
    for i in range(5):
        real_list = []
        for j in range(18):
            ori_path = retrieval_list[i*5 + j]
            real_path = './images/dataset/' + ori_path.split('/')[-2] + '/' + ori_path.split('/')[-1][:-4] + '.png'
            name = ori_path.split('/')[-2] + '/' + ori_path.split('/')[-1][:-4]
            real_list.append((real_path,name))
        real_list_set.append(real_list)

    # convert to dic for json dump
    path_dic = []
    for i in range(90):
        tmp = {}
        ori_path = retrieval_list[i * 5 + j]
        real_path = './images/dataset/' + ori_path.split('/')[-2] + '/' + ori_path.split('/')[-1][:-4] + '.png'
        name = ori_path.split('/')[-2] + '/' + ori_path.split('/')[-1][:-4]
        tmp['path'] = real_path
        tmp['name'] = name
        path_dic.append(tmp)

    return real_list_set, path_dic

def retrieval(net, sketch_path, dataset):
    sketch_src = Image.open(sketch_path).convert('RGB')
    feat_s = extract_feat_sketch(net, sketch_src)
    feat_photo_path = '../SketchTriplet/out_feat/flickr15k_1904041458/feat.npz'
    feat_photo = np.load(feat_photo_path)

    feat_p = feat_photo['feat']
    cls_name_p = feat_photo['cls_name']
    cls_num_p = feat_photo['cls_num']
    path_p = feat_photo['path']
    name_p = feat_photo['name']

    dist_l2 = np.sqrt(np.sum(np.square(feat_s - feat_p), 1))
    order = np.argsort(dist_l2)
    order_path_p = path_p[order]

    return get_real_path(order_path_p)