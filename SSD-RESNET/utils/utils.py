import numpy 
import stack

def point-form(bboxes):

    tl = bboxes[:,:2] - bboxes[:,2:]/2 #Top Left Corner
    br = bboxes[:,:2] + bboxes[:,2:]/2 #Bottom Right Corner

    return np.concatenate([tl,br],axis=1)



def detect_collate(batches):
    imgs =[]
    targets =[]

    for sample in batches:
        imgs.append(sample[0])
        targets.append(sample[1])

    stacked_images= torch.stack(imgs)
    np_imgs = np.array(targets)
    return stacked_images,np_imgs


def bbox_iou(box_a, box_b):
   
    m = box_a.shape[0]
    n = box_b.shape[0]

    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.maximum(br-tl, 0)
    
    inner = wh[:, :, 0]*wh[:, :, 1]

    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]

    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]

    a = a[:, None]
    b = b[None, :]

    

    return inner / (a+b-inner)

def nms(boxes, score, threshold=0.4):
   

    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break

        sort_ids = sort_ids[:-1]
        box = boxes[i].reshape(1, 4)
        ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)

        sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])

    return pick


