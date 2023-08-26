
import mindspore
import mindspore.ops as ops
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from data.transforms import train_aug, val_aug
from utils.box_list import BoxList

class ImageList:
    def __init__(self, img, ori_size, id):
        self.img = img
        self.ori_size = ori_size
        self.id = id

    def to_device(self, device):
        # 将图片转移到指定设备
        if isinstance(self.img, mindspore.Tensor):
            self.img = self.img.to(device)

    def __repr__(self):
        if isinstance(self.img, mindspore.Tensor):
            s = f'\nimg: {self.img.shape}, {self.img.dtype}, {self.img.device}, need_grad: {self.img.requires_grad}'
        elif isinstance(self.img, Image.Image):
            s = f'\nimg: {type(self.img)}'
        else:
            raise TypeError(f'Unrecognized img type, got {type(self.img)}.')
        for k, v in vars(self).items():
            if k != 'img':
                s += f'\n{k}: {v}'
        return s + '\n'

class COCODataset:
    def __init__(self, cfg, valing):
        self.cfg = cfg
        self.valing = valing

        img_path = cfg.train_imgs if not valing else cfg.val_imgs
        ann_file = cfg.train_ann if not valing else cfg.val_ann
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.getImgIds())

        if not valing:
            self.ids = [img_id for img_id in self.ids if self.has_valid_annotation(self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)))]
            self.aug = train_aug
        else:
            self.aug = val_aug

        self.to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.to_category_id = {v: k for k, v in self.to_contiguous_id.items()}
        self.id_img_map = {k: v for k, v in enumerate(self.ids)}

    def get_img_info(self, index):
        img_id = self.id_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    @staticmethod
    def has_valid_annotation(anno):
        if len(anno) == 0:
            return False
        if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
            return False
        return True

    def __getitem__(self, index):
        img_id = self.id_img_map[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(img_path).convert("RGB")

        anno = [aa for aa in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) if aa["iscrowd"] == 0]
        box = np.array([aa["bbox"] for aa in anno], dtype=np.float32).reshape(-1, 4)
        category = [self.to_contiguous_id[aa["category_id"]] for aa in anno]

        img_list = ImageList(img, ori_size=img.size, id=index)
        box_list = BoxList(box, img.size, 'x1y1wh', label=mindspore.Tensor(category, dtype=mindspore.int32))
        box_list.convert_mode('x1y1x2y2')
        box_list.clip_to_image(remove_empty=True)

        img_list, box_list = self.aug(img_list, box_list, self.cfg)

        return img_list, box_list

    def __len__(self):
        if not self.valing or self.cfg.val_num == -1:
            return len(self.ids)
        else:
            return min(self.cfg.val_num, len(self.ids))

