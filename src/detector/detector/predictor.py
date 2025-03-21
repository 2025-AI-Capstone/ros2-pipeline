from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
'''
detection 결과 중에서 pose estimation을 수행할 객체를 선택하는 기준을 정의하는 클래스입니다.
'''
class SelectivePosePredictor(PosePredictor):
    '''
    Args:
        cfg (dict): 모델 설정을 저장하는 딕셔너리
        overrides (list): cfg 설정을 덮어쓸 설정 목록
        _callbacks (list): 콜백 함수 목록
        selection_criteria (str): 객체 선택 기준
            if selection_criteria = 'top'
                top_num is the number of objects which are extracted keypoints
            elif selection_criteria = 'conf_threshold'
    '''
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, selection_criteria='conf_threshold', threshold=0.5, top_num = 5, center_area_width_ratio=0.5, center_area_height_ratio=0.5):
        super().__init__(cfg, overrides, _callbacks)
        self.selection_criteria = selection_criteria
        self.threshold = threshold
        self.center_area_width_ratio = center_area_width_ratio
        self.center_area_height_ratio = center_area_height_ratio

    def calculate_center_distance(self, boxes, img_shape):
        """
        이미지 중심으로부터의 거리를 계산합니다.
        
        Args:
            boxes (Tensor): 바운딩 박스 좌표 (x1, y1, x2, y2)
            img_shape (tuple): 이미지 크기 (height, width)
        
        Returns:
            Tensor: 각 박스의 중심점과 이미지 중심 간의 거리
        """
        # 이미지 중심점 계산
        img_center_x = img_shape[1] / 2
        img_center_y = img_shape[0] / 2
        
        # 박스의 중심점 계산
        box_center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        box_center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        
        # 유클리디안 거리 계산
        distances = torch.sqrt(
            (box_center_x - img_center_x) ** 2 + 
            (box_center_y - img_center_y) ** 2
        )
        
        # 거리를 이미지 대각선 길이로 정규화
        diagonal = torch.sqrt(torch.tensor(img_shape[0] ** 2 + img_shape[1] ** 2))
        normalized_distances = distances / diagonal
        
        return normalized_distances

    '''
    Args:
        pred (Tensor): 모델의 출력 결과
        selection_criteria (str): 객체 선택 기준
    Returns:
        Tensor: 선택된 객체의 인덱스
    '''
    def select_pose_candidates(self, pred, orig_img_shape, selection_criteria):
        '''
        if selection_criteria == 'top':
            conf_sorted_idx = torch.argsort(pred[:, 5], descending=True)
            return conf_sorted_idx[:min(self.top_num, len(conf_sorted_idx))]
        '''
        
        if selection_criteria == 'conf_threshold':
            # 특정 confidence 이상인 것들
            return torch.where(pred[:, 4] > self.threshold)[0]

        elif selection_criteria == 'center_area':
            # 이미지 크기 추출
            img_h, img_w, _ = orig_img_shape
            
            # 이미지 중심 좌표 계산
            img_center_x, img_center_y = img_w / 2, img_h / 2
            
            # 비례로 중앙 영역 경계 계산 (이미지 크기의 일정 비율)
            # self.center_area_width_ratio와 self.center_area_height_ratio는 0~1 사이 값
            half_width = (self.center_area_width_ratio * img_w) / 2
            half_height = (self.center_area_height_ratio * img_h) / 2
            
            min_x = img_center_x - half_width
            max_x = img_center_x + half_width
            min_y = img_center_y - half_height
            max_y = img_center_y + half_height
            
            # 각 객체의 중심점 계산
            boxes = pred[:, :4]  # [x1, y1, x2, y2] 형식
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
            
            # 중앙 영역 내에 있는 객체 인덱스 찾기
            center_area_mask = (centers_x >= min_x) & (centers_x <= max_x) & (centers_y >= min_y) & (centers_y <= max_y)
            
            return torch.where(center_area_mask)[0]
        return torch.arange(len(pred))  # 기본값: 모든 객체

    '''
    Args:
        preds (Tensor): 모델의 출력 결과
        img (Tensor): 입력 이미지
        orig_imgs (list): 원본 이미지
    Returns:
        list: Detection 결과
    '''
    def postprocess(self, preds, img, orig_imgs):
        """Return detection results with selective pose estimation."""
        # NMS 수행
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 바운딩 박스 스케일 조정
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            
            if len(pred) > 0:
                # 동적으로 선택 기준 결정 - 객체가 5개 이하면 conf_threshold, 그 이상이면 center_area
                current_criteria = 'conf_threshold' if len(pred) <= 5 else 'center_area'
                
                # 선택된 기준으로 객체 선택
                pose_indices = self.select_pose_candidates(pred, orig_img.shape, current_criteria)
                
                # 선택된 객체만 pose estimation
                pred_kpts = torch.zeros((len(pred), *self.model.kpt_shape), device=pred.device)
                if len(pose_indices) > 0:
                    selected_pose = pred[pose_indices, 6:].view(-1, *self.model.kpt_shape)
                    selected_pose = ops.scale_coords(img.shape[2:], selected_pose, orig_img.shape)
                    pred_kpts[pose_indices] = selected_pose
            else:
                pred_kpts = pred[:, 6:]

            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results