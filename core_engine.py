import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # <--- 新增这一行，强行镇压底层库冲突
import pickle
import sys
import time
import numpy as np
import cv2
import torch
from PIL import ImageFont, Image, ImageDraw
from PyQt5.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal
from skimage.feature import local_binary_pattern
from torch.autograd import Variable
from ssd_net_vgg import *
from voc0712 import *
from torchvision import transforms
from model_v2 import MobileNetV2
import matplotlib.font_manager as fm
import utils
from sigjiansuobasic import find_most_similar
from detection import *
from fatigue_landmark_detector import LandmarkFatigueTracker
from pretrained_eye_mouth import PretrainedEyeMouthClassifier
# import pyttsx3
import matplotlib
matplotlib.use('Agg')

ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'-----{"gpu" if torch.cuda.is_available() else "cpu"} mode-----')


class ModelRepository:
    """Load heavyweight models once and share them across sequential modes."""
    _mutex = QMutex()
    _ssd_model = None
    _face_model = None
    _face_detector = None
    _eye_mouth_classifier = None
    _eye_mouth_classifier_key = None

    @classmethod
    def ssd_model(cls):
        with QMutexLocker(cls._mutex):
            if cls._ssd_model is None:
                net = SSD()
                net.train(mode=False)
                state = torch.load(os.path.join(ROOT, "weights", "ssd_voc_120000.pth"), map_location=device)
                state = {key.replace("module.", ""): value for key, value in state.items()}
                net.load_state_dict(state)
                cls._ssd_model = net.to(device).eval()
            return cls._ssd_model

    @classmethod
    def face_model(cls):
        with QMutexLocker(cls._mutex):
            if cls._face_model is None:
                model = MobileNetV2(num_classes=2)
                state = torch.load(os.path.join(ROOT, "faceopenset_mobilenet123.pth"), map_location=device)
                current = model.state_dict()
                current.update({key: value for key, value in state.items()
                                if key in current and value.shape == current[key].shape})
                model.load_state_dict(current)
                cls._face_model = model.to(device).eval()
            return cls._face_model

    @classmethod
    def face_detector(cls):
        with QMutexLocker(cls._mutex):
            if cls._face_detector is None:
                cls._face_detector = cv2.dnn.readNetFromCaffe(
                    os.path.join(ROOT, "deploy.prototxt"),
                    os.path.join(ROOT, "res10_300x300_ssd_iter_140000.caffemodel"))
            return cls._face_detector

    @classmethod
    def eye_mouth_classifier(cls, eye_weights, mouth_weights, model_device="cpu"):
        key = (
            os.path.abspath(eye_weights),
            os.path.abspath(mouth_weights),
            model_device,
        )
        with QMutexLocker(cls._mutex):
            if cls._eye_mouth_classifier is None:
                cls._eye_mouth_classifier = PretrainedEyeMouthClassifier(
                    eye_weights=key[0],
                    mouth_weights=key[1],
                    device=model_device,
                )
                cls._eye_mouth_classifier_key = key
            elif cls._eye_mouth_classifier_key != key:
                raise RuntimeError("眼嘴模型已使用不同的权重路径初始化")
            return cls._eye_mouth_classifier


def init_tts_engine():
    return None  # <--- 新增这一行！直接跳过语音引擎初始化
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        print(f"初始化语音引擎失败: {e}")
        return None


def init_model():
    return ModelRepository.face_model()


def init_eye_mouth_classifier():
    eye_weights = os.environ.get(
        "FATIGUE_EYE_WEIGHTS",
        os.path.join(ROOT, "weights", "best_eye_classifier.pt"),
    )
    mouth_weights = os.environ.get(
        "FATIGUE_MOUTH_WEIGHTS",
        os.path.join(ROOT, "weights", "best_mouth_classifier.pt"),
    )
    model_device = os.environ.get("FATIGUE_MODEL_DEVICE", "cpu")

    if not os.path.isfile(eye_weights) or not os.path.isfile(mouth_weights):
        print(
            "未配置预训练眼嘴模型，使用EAR/MAR。请设置 "
            "FATIGUE_EYE_WEIGHTS 和 FATIGUE_MOUTH_WEIGHTS。"
        )
        return None

    try:
        classifier = ModelRepository.eye_mouth_classifier(
            eye_weights,
            mouth_weights,
            model_device,
        )
        print(f"预训练眼嘴模型已加载: {model_device}")
        return classifier
    except Exception as exc:
        print(f"预训练眼嘴模型加载失败，使用EAR/MAR: {exc}")
        return None


def init_landmark_tracker():
    try:
        return LandmarkFatigueTracker(
            state_classifier=init_eye_mouth_classifier()
        )
    except Exception as exc:
        print(f"关键点疲劳检测初始化失败，回退到SSD检测: {exc}")
        return None


class FatigueEvaluator:
    """Fatigue evaluator based on sliding-window temporal features."""

    def __init__(self):
        self.weights = {
            'perclos': 0.4,
            'blink': 0.2,
            'yawn': 0.25,
            'eye_closure': 0.3,
            'mouth_duration': 0.2,
        }
        self.fatigue_threshold = 0.4
        self.perclos_threshold = 0.2
        self.blink_threshold = (0.2, 0.8)
        self.yawn_threshold = 2
        self.eye_closure_threshold = 1.5
        self.yawn_duration_threshold = 2.0

    def evaluate(self, perclos, blink_rate, yawn_count, window_features=None):
        """Return a fatigue score using recent 30-60s temporal features."""
        window_features = window_features or {}
        perclos = window_features.get("perclos", perclos)
        blink_rate = window_features.get("blink_rate", blink_rate)
        yawn_count = window_features.get("yawn_count", yawn_count)
        longest_eye_closure = max(
            window_features.get("longest_eye_closure", 0.0),
            window_features.get("current_eye_closure", 0.0),
        )
        max_yawn_duration = max(
            window_features.get("max_yawn_duration", 0.0),
            window_features.get("current_yawn_duration", 0.0),
        )

        perclos_score = min(perclos / self.perclos_threshold, 1.0)

        if blink_rate < self.blink_threshold[0]:
            blink_score = (self.blink_threshold[0] - blink_rate) / self.blink_threshold[0]
        elif blink_rate > self.blink_threshold[1]:
            blink_score = (blink_rate - self.blink_threshold[1]) / (2 - self.blink_threshold[1])
        else:
            blink_score = 0

        yawn_score = min(yawn_count / self.yawn_threshold, 1.0)
        eye_closure_score = min(longest_eye_closure / self.eye_closure_threshold, 1.0)
        mouth_duration_score = min(max_yawn_duration / self.yawn_duration_threshold, 1.0)

        total_score = (self.weights['perclos'] * perclos_score +
                       self.weights['blink'] * blink_score +
                       self.weights['yawn'] * yawn_score +
                       self.weights['eye_closure'] * eye_closure_score +
                       self.weights['mouth_duration'] * mouth_duration_score)

        is_fatigue = (
            total_score >= self.fatigue_threshold
            or longest_eye_closure >= self.eye_closure_threshold
            or max_yawn_duration >= self.yawn_duration_threshold
        )

        return {
            'fatigue': is_fatigue,
            'score': total_score,
            'perclos': perclos,
            'blink_rate': blink_rate,
            'yawn_count': yawn_count,
            'longest_eye_closure': longest_eye_closure,
            'max_yawn_duration': max_yawn_duration,
            'window_features': window_features,
            'weights': self.weights
        }


class BaseDetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_update_signal = pyqtSignal(dict)

    def __init__(self, needs_ssd=True):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.net = ModelRepository.ssd_model() if needs_ssd else None
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.img_mean = (104.0, 117.0, 123.0)

        try:
            font_path = fm.findfont(fm.FontProperties(family='SimHei'))
            self.font = ImageFont.truetype(font_path, 20)
            self.small_font = ImageFont.truetype(font_path, 14)
        except:
            try:
                self.font = ImageFont.truetype("msyh.ttc", 20)
                self.small_font = ImageFont.truetype("msyh.ttc", 14)
            except:
                self.font = ImageFont.load_default()
                self.small_font = ImageFont.load_default()

    def cv2_add_chinese_text(self, img, text, position, textColor=(0, 255, 0), textSize=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        use_font = self.small_font if textSize <= 14 else self.font
        draw.text(position, text, textColor, font=use_font)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def stop(self):
        self._run_flag = False
        self.requestInterruption()
        if not self.wait(5000):
            self.status_update_signal.emit({"error": "检测线程未能在 5 秒内停止"})

    def _capture_loop(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.status_update_signal.emit({"error": "无法打开摄像头 0"})
                return
            while self._run_flag and not self.isInterruptionRequested():
                ret, image = self.cap.read()
                if not ret:
                    self.msleep(20)
                    continue
                started = time.perf_counter()
                processed, status = self.process_frame(image)
                status["fps"] = 1.0 / max(time.perf_counter() - started, 1e-6)
                self.change_pixmap_signal.emit(processed)
                self.status_update_signal.emit(status)
        except Exception as exc:
            self.status_update_signal.emit({"error": str(exc)})
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class ContinuousFatigueDetectionThread(BaseDetectionThread):
    def __init__(self):
        super().__init__()
        self.eye_closed_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.last_blink_time = None
        self.last_yawn_time = None
        self.blink_detected = False
        self.yawn_detected = False
        self.evaluator = FatigueEvaluator()
        self.start_time = time.time()
        self.tts_engine = init_tts_engine()
        self.landmark_tracker = init_landmark_tracker()

    def run(self):
        self._capture_loop()

    def process_frame(self, img):
        if self.landmark_tracker is not None:
            try:
                landmark_status = self.landmark_tracker.process_frame(img, update=True, draw=True)
                if landmark_status.get("face_found"):
                    perclos = landmark_status["perclos"]
                    blink_rate = landmark_status["blink_rate"]
                    yawn_count = landmark_status["yawn_count"]
                    window_features = landmark_status.get("window_features", {})
                    result = self.evaluator.evaluate(perclos, blink_rate, yawn_count, window_features)
                    status_info = {
                        "perclos": perclos,
                        "blink_count": landmark_status["blink_count"],
                        "blink_rate": blink_rate,
                        "yawn_count": yawn_count,
                        "score": result['score'],
                        "fatigue": result['fatigue'],
                        "longest_eye_closure": result["longest_eye_closure"],
                        "max_yawn_duration": result["max_yawn_duration"],
                        "window_features": window_features,
                        "eye_state": landmark_status["eye_state"],
                        "mouth_state": landmark_status["mouth_state"],
                        "fatigue_features": landmark_status["fatigue_features"],
                        "ear": landmark_status["ear"],
                        "mar": landmark_status["mar"],
                        "eye_closed_probability": landmark_status.get("eye_closed_probability"),
                        "mouth_open_probability": landmark_status.get("mouth_open_probability"),
                        "model_latency_ms": landmark_status.get("model_latency_ms"),
                        "detector": landmark_status.get("detector", "landmark_rule"),
                    }
                    if result['fatigue']:
                        img = self.cv2_add_chinese_text(img, "疲劳警告!", (50, 120), (0, 0, 255), 30)
                    return img, status_info

                return img, {
                    "perclos": landmark_status["perclos"],
                    "blink_count": landmark_status["blink_count"],
                    "blink_rate": 0,
                    "yawn_count": landmark_status["yawn_count"],
                    "window_features": landmark_status.get("window_features", {}),
                    "score": 0,
                    "fatigue": False,
                    "eye_state": landmark_status["eye_state"],
                    "mouth_state": landmark_status["mouth_state"],
                    "fatigue_features": ["未检测到人脸"],
                    "detector": landmark_status.get("detector", "landmark_rule"),
                }
            except Exception as exc:
                print(f"关键点疲劳检测失败，回退到SSD检测: {exc}")
                self.landmark_tracker = None

        eye_closed = False
        mouth_open = False

        # SSD疲劳检测
        x = cv2.resize(img.copy(), (300, 300)).astype(np.float32)
        x -= self.img_mean
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0)).to(device)

        with torch.no_grad():
            y = self.net(xx)

        softmax = nn.Softmax(dim=-1)
        detect = Detect.apply
        priors = utils.default_prior_box()

        loc, conf = y
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        detections = detect(
            loc.view(loc.size(0), -1, 4),
            softmax(conf.view(conf.size(0), -1, config.class_num)),
            torch.cat([o.view(-1, 4) for o in priors], 0),
            config.class_num,
            200,
            0.7,
            0.45
        ).data

        labels = VOC_CLASSES
        scale = torch.Tensor(img.shape[1::-1]).repeat(2).to(detections.device)

        # 绘制疲劳检测结果
        fatigue_detections = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.4:
                score = detections[0, i, j, 0]
                label_name = labels[i - 1]

                if label_name == 'closed_eye':
                    eye_closed = True
                if label_name == 'open_mouth':
                    mouth_open = True

                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                x1, y1, x2, y2 = map(int, pt)

                color = (214, 39, 40) if i == 0 else (23, 190, 207) if i == 1 else (188, 189, 34)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{label_name}:{score:.2f}',
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

                fatigue_detections.append((label_name, score))
                j += 1

        # 更新疲劳检测指标
        self.total_frames += 1
        if eye_closed:
            self.eye_closed_frames += 1

        # 检测眨眼
        if eye_closed:
            self.blink_detected = True
        elif self.blink_detected:  # 从闭眼到睁眼，完成一次眨眼
            self.blink_count += 1
            self.blink_detected = False

        # 检测哈欠
        if mouth_open:
            if not self.yawn_detected:
                self.yawn_detected = True
                self.yawn_count += 1
        else:
            self.yawn_detected = False

        # 计算实时疲劳状态
        elapsed = max(1, time.time() - self.start_time)
        perclos = self.eye_closed_frames / self.total_frames
        blink_rate = self.blink_count / elapsed
        yawn_count = self.yawn_count

        result = self.evaluator.evaluate(perclos, blink_rate, yawn_count)

        # 准备状态信息
        status_info = {
            "perclos": perclos,
            "blink_count": self.blink_count,
            "blink_rate": blink_rate,
            "yawn_count": yawn_count,
            "score": result['score'],
            "fatigue": result['fatigue'],
            "eye_state": "闭合" if eye_closed else "正常",
            "mouth_state": "张开" if mouth_open else "正常",
            "fatigue_features": fatigue_detections[:4] if fatigue_detections else ["未检测到疲劳特征"]
        }

        # 疲劳状态显示
        if result['fatigue']:
            img = self.cv2_add_chinese_text(img, "疲劳警告!", (50, 50), (0, 0, 255), 30)

        return img, status_info



class FaceRecognitionThread(BaseDetectionThread):
    def __init__(self):
        super().__init__(needs_ssd=False)
        self.face_recognizer = init_model()
        self.face_recognizer.eval()
        self.face_detector = ModelRepository.face_detector()
        self.database_folder = os.path.join(ROOT, "dataset", "face_features")
        self.face_cache = {}
        self.last_recognition = time.time()
        self.tts_engine = init_tts_engine()
        self.last_recognized_identity = None  # 新增：记录上次识别的身份

        self.database_file = os.path.join(ROOT, "traditional_features.pkl")

        self.database_features = self.load_or_build_database(os.path.join(ROOT, "dataset", "data"))

    def extract_lbp_features(self, gray_img):
        """提取LBP特征"""
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # 归一化
        return hist

    def extract_haar_features(self, gray_img):
        """安全的Haar特征提取"""
        try:
            # 确保输入尺寸有效
            h, w = gray_img.shape
            if h < 30 or w < 30:  # 最小支持30x30的窗口
                raise ValueError(f"图像尺寸{h}x{w}过小，至少需要30x30")

            # 计算积分图像 (添加1像素边框)
            integral = cv2.integral(gray_img)

            features = []
            min_window_size = 20  # 减小最小窗口尺寸

            # 确保特征坐标不越界
            for win_w in range(min_window_size, w // 2, 10):
                for win_h in range(min_window_size, h // 2, 10):
                    for x in range(0, w - win_w, win_w // 2):
                        for y in range(0, h - win_h, win_h // 2):
                            # 水平双矩形特征
                            if x + 2 * win_w <= w and y + win_h <= h:
                                white = integral[y, x] + integral[y + win_h, x + win_w] - \
                                        integral[y, x + win_w] - integral[y + win_h, x]
                                black = integral[y, x + win_w] + integral[y + win_h, x + 2 * win_w] - \
                                        integral[y, x + 2 * win_w] - integral[y + win_h, x + win_w]
                                features.append(white - black)

                            # 垂直双矩形特征
                            if x + win_w <= w and y + 2 * win_h <= h:
                                white = integral[y, x] + integral[y + win_h, x + win_w] - \
                                        integral[y, x + win_w] - integral[y + win_h, x]
                                black = integral[y + win_h, x] + integral[y + 2 * win_h, x + win_w] - \
                                        integral[y + win_h, x + win_w] - integral[y + 2 * win_h, x]
                                features.append(white - black)

            features = np.array(features)
            return features / (np.linalg.norm(features) + 1e-6)  # 归一化

        except Exception as e:
            print(f"Haar特征提取失败: {e}")
            return None

    def match_with_database(self, lbp, haar):
        """与传统特征数据库匹配"""
        min_dist = float('inf')
        best_match = "Unknown"
        self.threshold = 0.6

        # 遍历数据库中的特征
        for name in self.database_features:
            db_lbp, db_haar = self.database_features[name]

            # 计算LBP特征距离
            lbp_dist = np.sum(np.abs(lbp - db_lbp))

            # 计算Haar特征距离
            haar_dist = np.sum(np.abs(haar[:len(db_haar)] - db_haar))

            # 组合距离
            total_dist = 0.5 * lbp_dist + 0.5 * haar_dist

            self.threshold = 25.5386
            if total_dist < min_dist and total_dist < self.threshold:
                min_dist = total_dist
                best_match = name

        return best_match

    def load_or_build_database(self, database_folder):
        """加载或构建特征数据库"""
        if os.path.exists(self.database_file):
            print("加载已有特征数据库...")
            with open(self.database_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("构建新的特征数据库...")
            features = self.build_traditional_database(database_folder)
            with open(self.database_file, 'wb') as f:
                pickle.dump(features, f)
            return features

    # def build_traditional_database(self, database_folder):
    #     """构建传统特征数据库"""
    #     database_features = {}
    #
    #     for person_name in os.listdir(database_folder):
    #         person_dir = os.path.join(database_folder, person_name)
    #         if os.path.isdir(person_dir):
    #             features = []
    #             for img_file in os.listdir(person_dir):
    #                 img_path = os.path.join(person_dir, img_file)
    #                 try:
    #                     img = cv2.imread(img_path)
    #                     if img is None:
    #                         continue
    #
    #                     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #                     gray = cv2.resize(gray, (224,224))
    #                     print(gray.shape)
    #                     # 提取特征
    #                     lbp = self.extract_lbp_features(gray)
    #                     haar = self.extract_haar_features(gray)
    #
    #                     features.append((lbp, haar))
    #                 except Exception as e:
    #                     print(f"处理 {img_path} 时出错: {e}")
    #                     continue
    #
    #             if features:  # 确保至少有一个有效特征
    #                 # 取多张图像的平均特征
    #                 avg_lbp = np.mean([f[0] for f in features], axis=0)
    #                 avg_haar = np.mean([f[1] for f in features], axis=0)
    #
    #                 database_features[person_name] = (avg_lbp, avg_haar)
    #     print(database_features)
    #
    #     return database_features

    # def build_traditional_database(self, database_folder):
    #     """构建传统特征数据库"""
    #     database_features = {}
    #
    #     print(f"开始扫描数据库目录: {database_folder}")
    #     person_list = os.listdir(database_folder)
    #     print(f"找到 {len(person_list)} 个子目录/文件")
    #
    #     for person_name in person_list:
    #         person_dir = os.path.join(database_folder, person_name)
    #         if not os.path.isdir(person_dir):
    #             print(f"跳过非目录项: {person_name}")
    #             continue
    #
    #         print(f"\n处理人物: {person_name}")
    #         features = []
    #         valid_count = 0
    #
    #         for img_file in os.listdir(person_dir):
    #             img_path = os.path.join(person_dir, img_file)
    #             try:
    #                 # 1. 读取图像
    #                 img = cv2.imread(img_path)
    #                 if img is None:
    #                     print(f"  警告: 无法读取 {img_file}")
    #                     continue
    #
    #                 # 2. 转为灰度图并调整尺寸
    #                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #                 gray = cv2.resize(gray, (224, 224))
    #
    #                 # 3. 提取特征
    #                 lbp = self.extract_lbp_features(gray)
    #                 haar = self.extract_haar_features(gray)
    #
    #                 if lbp is None or haar is None:
    #                     print(f"  警告: {img_file} 特征提取失败")
    #                     continue
    #
    #                 features.append((lbp, haar))
    #                 valid_count += 1
    #                 print(f"  √ 处理成功: {img_file}")
    #
    #             except Exception as e:
    #                 print(f"  处理 {img_file} 时出错: {str(e)}")
    #                 continue
    #
    #         # 保存该人物的特征
    #         if features:
    #             avg_lbp = np.mean([f[0] for f in features], axis=0)
    #             avg_haar = np.mean([f[1] for f in features], axis=0)
    #             database_features[person_name] = (avg_lbp, avg_haar)
    #             print(f"√ 成功添加 {person_name}: {valid_count} 张有效图像")
    #         else:
    #             print(f"× 跳过 {person_name}: 无有效图像")
    #
    #     print("\n数据库构建完成，统计结果:")
    #     print(f"总人物数: {len(database_features)}")
    #     print("人物列表:", list(database_features.keys()))
    #     return database_features

    def build_traditional_database(self, database_folder):
        """改进版：支持平面文件结构"""
        database_features = {}
        name_features = {}  # 临时按人名存储特征

        print(f"扫描目录: {database_folder}")

        for img_file in os.listdir(database_folder):
            img_path = os.path.join(database_folder, img_file)
            if not os.path.isfile(img_path):
                continue

            try:
                # 从文件名提取人名 (如"chenliqing_1.png" -> "chenliqing")
                person_name = img_file.split('_')[0]

                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取: {img_file}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (224, 224))

                lbp = self.extract_lbp_features(gray)
                haar = self.extract_haar_features(gray)

                if person_name not in name_features:
                    name_features[person_name] = []
                name_features[person_name].append((lbp, haar))
                print(f"√ 处理成功: {img_file} -> {person_name}")

            except Exception as e:
                print(f"处理 {img_file} 出错: {e}")

        # 计算平均特征
        for name, features in name_features.items():
            avg_lbp = np.mean([f[0] for f in features], axis=0)
            avg_haar = np.mean([f[1] for f in features], axis=0)
            database_features[name] = (avg_lbp, avg_haar)

        print(f"\n构建完成，共 {len(database_features)} 个人物")
        return database_features
    def prepare_face_input(self, face_img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(face_img).unsqueeze(0).to(device)

    def run(self):
        self._capture_loop()

    def process_frame(self, img):
        # 人脸检测
        best_confidence = 0.0
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        face_detections = self.face_detector.forward()

        # 处理人脸检测结果
        detected_faces = []
        face_detected = False

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence > 0.5:
                best_confidence = max(best_confidence, float(confidence))
                face_detected = True
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if time.time() - self.last_recognition > 1.0:
                    try:
                        face_img = img[y1:y2, x1:x2]
                        if face_img.size > 0:


                            face_key = f"{x1}_{y1}_{x2}_{y2}"

                            if face_key not in self.face_cache:
                                # 使用传统方法进行人脸特征提取和识别
                                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                                gray_face = cv2.resize(gray_face,(224,224))
                                print(gray_face.shape)

                                # 1. LBP特征提取 (局部二值模式)
                                lbp = self.extract_lbp_features(gray_face)
                                print(lbp)

                                print("*"*100)
                                # 2. Haar-like特征提取
                                haar = self.extract_haar_features(gray_face)


                                # 3. 与数据库中的特征进行匹配
                                identity = self.match_with_database(lbp, haar)

                                self.face_cache[face_key] = (identity, time.time())
                                self.last_recognized_identity = identity

                            if face_key in self.face_cache:
                                identity, _ = self.face_cache[face_key]
                                detected_faces.append(identity)
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                img = self.cv2_add_chinese_text(img, f"{identity}",
                                                                (x1, y1 - 30), (0, 255, 0), 16)
                                cv2.putText(img, f"{confidence:.2f}",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Face recognition error: {e}")

        if time.time() - self.last_recognition > 1.0:
            self.last_recognition = time.time()

        # 准备状态信息 - 如果没有检测到新人脸，但仍保持上次识别的身份
        if not detected_faces and self.last_recognized_identity:
            detected_faces = [self.last_recognized_identity]

        status_info = {
            "detected_faces": detected_faces[:3] if detected_faces else ["未检测到人脸"],
            "last_recognized_identity": self.last_recognized_identity,
            "confidence": best_confidence,
        }

        return img, status_info


class FatigueCheckInThread(BaseDetectionThread):
    def __init__(self):
        super().__init__()
        self.face_recognizer = init_model()
        self.face_recognizer.eval()
        self.face_detector = ModelRepository.face_detector()
        self.database_folder = os.path.join(ROOT, "dataset", "face_features")
        self.face_cache = {}
        self.last_recognition = time.time()
        self.eye_closed_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.last_blink_time = None
        self.last_yawn_time = None
        self.blink_detected = False
        self.yawn_detected = False
        self.evaluator = FatigueEvaluator()
        self.current_identity = None
        self.detection_start_time = None
        self.evaluation_duration = 5
        self.is_evaluating = False
        self.evaluation_result = None
        self.last_face_time = None
        self.tts_engine = init_tts_engine()
        self.database_file = os.path.join(ROOT, "traditional_features.pkl")
        self.landmark_tracker = init_landmark_tracker()
        self.latest_window_features = {}

        self.database_features = self.load_or_build_database(os.path.join(ROOT, "dataset", "data"))

    def prepare_face_input(self, face_img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(face_img).unsqueeze(0).to(device)

    def extract_lbp_features(self, gray_img):
        """提取LBP特征"""
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # 归一化
        return hist

    def extract_haar_features(self, gray_img):
        """安全的Haar特征提取"""
        try:
            # 确保输入尺寸有效
            h, w = gray_img.shape
            if h < 30 or w < 30:  # 最小支持30x30的窗口
                raise ValueError(f"图像尺寸{h}x{w}过小，至少需要30x30")

            # 计算积分图像 (添加1像素边框)
            integral = cv2.integral(gray_img)

            features = []
            min_window_size = 20  # 减小最小窗口尺寸

            # 确保特征坐标不越界
            for win_w in range(min_window_size, w // 2, 10):
                for win_h in range(min_window_size, h // 2, 10):
                    for x in range(0, w - win_w, win_w // 2):
                        for y in range(0, h - win_h, win_h // 2):
                            # 水平双矩形特征
                            if x + 2 * win_w <= w and y + win_h <= h:
                                white = integral[y, x] + integral[y + win_h, x + win_w] - \
                                        integral[y, x + win_w] - integral[y + win_h, x]
                                black = integral[y, x + win_w] + integral[y + win_h, x + 2 * win_w] - \
                                        integral[y, x + 2 * win_w] - integral[y + win_h, x + win_w]
                                features.append(white - black)

                            # 垂直双矩形特征
                            if x + win_w <= w and y + 2 * win_h <= h:
                                white = integral[y, x] + integral[y + win_h, x + win_w] - \
                                        integral[y, x + win_w] - integral[y + win_h, x]
                                black = integral[y + win_h, x] + integral[y + 2 * win_h, x + win_w] - \
                                        integral[y + win_h, x + win_w] - integral[y + 2 * win_h, x]
                                features.append(white - black)

            features = np.array(features)
            return features / (np.linalg.norm(features) + 1e-6)  # 归一化

        except Exception as e:
            print(f"Haar特征提取失败: {e}")
            return None

    def match_with_database(self, lbp, haar):
        """与传统特征数据库匹配"""
        min_dist = float('inf')
        best_match = "Unknown"
        self.threshold = 0.6

        # 遍历数据库中的特征
        for name in self.database_features:
            db_lbp, db_haar = self.database_features[name]

            # 计算LBP特征距离
            lbp_dist = np.sum(np.abs(lbp - db_lbp))

            # 计算Haar特征距离
            haar_dist = np.sum(np.abs(haar[:len(db_haar)] - db_haar))

            # 组合距离
            total_dist = 0.5 * lbp_dist + 0.5 * haar_dist

            self.threshold = 25.5386
            if total_dist < min_dist and total_dist < self.threshold:
                min_dist = total_dist
                best_match = name

        return best_match

    def load_or_build_database(self, database_folder):
        """加载或构建特征数据库"""
        if os.path.exists(self.database_file):
            print("加载已有特征数据库...")
            with open(self.database_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("构建新的特征数据库...")
            features = self.build_traditional_database(database_folder)
            with open(self.database_file, 'wb') as f:
                pickle.dump(features, f)
            return features
    def build_traditional_database(self, database_folder):
        """改进版：支持平面文件结构"""
        database_features = {}
        name_features = {}  # 临时按人名存储特征

        print(f"扫描目录: {database_folder}")

        for img_file in os.listdir(database_folder):
            img_path = os.path.join(database_folder, img_file)
            if not os.path.isfile(img_path):
                continue

            try:
                # 从文件名提取人名 (如"chenliqing_1.png" -> "chenliqing")
                person_name = img_file.split('_')[0]

                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取: {img_file}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (224, 224))

                lbp = self.extract_lbp_features(gray)
                haar = self.extract_haar_features(gray)

                if person_name not in name_features:
                    name_features[person_name] = []
                name_features[person_name].append((lbp, haar))
                print(f"√ 处理成功: {img_file} -> {person_name}")

            except Exception as e:
                print(f"处理 {img_file} 出错: {e}")

        # 计算平均特征
        for name, features in name_features.items():
            avg_lbp = np.mean([f[0] for f in features], axis=0)
            avg_haar = np.mean([f[1] for f in features], axis=0)
            database_features[name] = (avg_lbp, avg_haar)

        print(f"\n构建完成，共 {len(database_features)} 个人物")
        return database_features

    def run(self):
        self._capture_loop()

    def process_frame(self, img):
        eye_closed = False
        mouth_open = False
        num_rec = 0
        best_confidence = 0.0

        # 人脸检测
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        face_detections = self.face_detector.forward()

        # 处理人脸检测结果
        detected_faces = []
        current_frame_identities = set()
        face_detected = False

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence > 0.5:
                best_confidence = max(best_confidence, float(confidence))
                face_detected = True
                self.update_face_detection()
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if time.time() - self.last_recognition > 1.0:
                    try:
                        face_img = img[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_key = f"{x1}_{y1}_{x2}_{y2}"
                            if face_key not in self.face_cache:
                                # 使用传统方法进行人脸特征提取和识别
                                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                                gray_face = cv2.resize(gray_face, (224, 224))
                                print(gray_face.shape)

                                # 1. LBP特征提取 (局部二值模式)
                                lbp = self.extract_lbp_features(gray_face)
                                print(lbp)

                                print("*" * 100)
                                # 2. Haar-like特征提取
                                haar = self.extract_haar_features(gray_face)

                                # 3. 与数据库中的特征进行匹配
                                identity = self.match_with_database(lbp, haar)

                                self.face_cache[face_key] = (identity, time.time())
                                self.last_recognized_identity = identity

                            if face_key in self.face_cache:
                                identity, _ = self.face_cache[face_key]
                                current_frame_identities.add(identity)
                                detected_faces.append(identity)

                                # 如果检测到新人脸且不在评估中，开始5秒评估
                                if not self.is_evaluating:
                                    self.start_evaluation(identity)

                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                img = self.cv2_add_chinese_text(img, f"{identity}",
                                                                (x1, y1 - 30), (0, 255, 0), 16)
                                cv2.putText(img, f"{confidence:.2f}",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Face recognition error: {e}")

        # 如果没有检测到人脸且正在评估，检查是否超时
        if not face_detected and self.is_evaluating:
            self.check_evaluation_status()

        if time.time() - self.last_recognition > 1.0:
            self.last_recognition = time.time()

        if self.landmark_tracker is not None:
            try:
                landmark_status = self.landmark_tracker.process_frame(
                    img, update=self.is_evaluating, draw=True
                )
                if self.is_evaluating:
                    self.eye_closed_frames = self.landmark_tracker.eye_closed_frames
                    self.total_frames = self.landmark_tracker.total_frames
                    self.blink_count = self.landmark_tracker.blink_count
                    self.yawn_count = self.landmark_tracker.yawn_count
                    self.latest_window_features = landmark_status.get("window_features", {})

                evaluation_status = self.check_evaluation_status()
                window_features = landmark_status.get("window_features", {})
                status_info = {
                    "identity": self.current_identity if self.current_identity else "未检测到",
                    "evaluating": self.is_evaluating,
                    "countdown": evaluation_status if isinstance(evaluation_status, (int, float)) else None,
                    "evaluation_result": self.evaluation_result,
                    "blink_count": self.blink_count,
                    "yawn_count": self.yawn_count,
                    "perclos": window_features.get("perclos", self.eye_closed_frames / max(1, self.total_frames)),
                    "blink_rate": window_features.get("blink_rate", 0),
                    "longest_eye_closure": window_features.get("longest_eye_closure", 0),
                    "max_yawn_duration": window_features.get("max_yawn_duration", 0),
                    "window_features": window_features,
                    "eye_state": landmark_status["eye_state"],
                    "mouth_state": landmark_status["mouth_state"],
                    "confidence": best_confidence,
                    "detected_faces": detected_faces[:3] if detected_faces else [""],
                    "fatigue_features": landmark_status.get("fatigue_features", ["未检测到疲劳特征"]),
                    "eye_closed_probability": landmark_status.get("eye_closed_probability"),
                    "mouth_open_probability": landmark_status.get("mouth_open_probability"),
                    "model_latency_ms": landmark_status.get("model_latency_ms"),
                    "detector": landmark_status.get("detector", "landmark_rule"),
                }
                return img, status_info
            except Exception as exc:
                print(f"关键点疲劳检测失败，回退到SSD检测: {exc}")
                self.landmark_tracker = None

        # SSD疲劳检测
        x = cv2.resize(img.copy(), (300, 300)).astype(np.float32)
        x -= self.img_mean
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0)).to(device)

        with torch.no_grad():
            y = self.net(xx)

        softmax = nn.Softmax(dim=-1)
        detect = Detect.apply
        priors = utils.default_prior_box()

        loc, conf = y
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        detections = detect(
            loc.view(loc.size(0), -1, 4),
            softmax(conf.view(conf.size(0), -1, config.class_num)),
            torch.cat([o.view(-1, 4) for o in priors], 0),
            config.class_num,
            200,
            0.7,
            0.45
        ).data

        labels = VOC_CLASSES
        scale = torch.Tensor(img.shape[1::-1]).repeat(2).to(detections.device)

        # 绘制疲劳检测结果
        fatigue_detections = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.4:
                score = detections[0, i, j, 0]
                label_name = labels[i - 1]

                if label_name == 'closed_eye':
                    eye_closed = True
                if label_name == 'open_mouth':
                    mouth_open = True

                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                x1, y1, x2, y2 = map(int, pt)

                color = (214, 39, 40) if i == 0 else (23, 190, 207) if i == 1 else (188, 189, 34)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{label_name}:{score:.2f}',
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

                fatigue_detections.append((label_name, score))
                j += 1
                num_rec += 1

        # 更新疲劳检测指标
        if self.is_evaluating:
            # 更新眼睛状态
            self.update_eye_state(eye_closed)

            # 检测眨眼
            if eye_closed:
                self.blink_detected = True
            elif self.blink_detected:  # 从闭眼到睁眼，完成一次眨眼
                self.record_blink()
                self.blink_detected = False

            # 检测哈欠
            if mouth_open:
                if not self.yawn_detected:
                    self.yawn_detected = True
                    self.record_yawn()
            else:
                self.yawn_detected = False

        # 检查评估状态
        evaluation_status = self.check_evaluation_status()

        # 准备状态信息
        status_info = {
            "identity": self.current_identity if self.current_identity else "未检测到",
            "evaluating": self.is_evaluating,
            "countdown": evaluation_status if isinstance(evaluation_status, (int, float)) else None,
            "evaluation_result": self.evaluation_result,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "perclos": self.eye_closed_frames / max(1, self.total_frames),
            "eye_state": "闭合" if eye_closed else "正常",
            "mouth_state": "张开" if mouth_open else "正常",
            "confidence": best_confidence,
            "detected_faces": detected_faces[:3] if detected_faces else [""],
            "fatigue_features": fatigue_detections[:4] if fatigue_detections else ["未检测到疲劳特征"]
        }

        return img, status_info

    def start_evaluation(self, identity):
        """开始5秒评估"""
        self.current_identity = identity
        self.detection_start_time = time.time()
        self.is_evaluating = True
        self.evaluation_result = None
        self.reset_metrics()
        print(f"开始5秒评估: {identity}")

    def reset_metrics(self):
        """重置评估指标"""
        if self.landmark_tracker is not None:
            self.landmark_tracker.reset()
        self.eye_closed_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.last_blink_time = None
        self.last_yawn_time = None
        self.latest_window_features = {}

    def update_face_detection(self):
        """更新人脸检测时间"""
        self.last_face_time = time.time()

    def update_eye_state(self, is_closed):
        """更新眼睛状态"""
        if not self.is_evaluating:
            return

        self.total_frames += 1
        if is_closed:
            self.eye_closed_frames += 1

    def record_blink(self):
        """记录眨眼"""
        if not self.is_evaluating:
            return

        current_time = time.time()
        # 避免重复记录眨眼
        if self.last_blink_time is None or (current_time - self.last_blink_time) > 0.2:
            self.blink_count += 1
            self.last_blink_time = current_time
            print(f"检测到眨眼，当前眨眼次数: {self.blink_count}")

    def record_yawn(self):
        """记录哈欠"""
        if not self.is_evaluating:
            return

        current_time = time.time()
        # 避免重复记录哈欠
        if self.last_yawn_time is None or (current_time - self.last_yawn_time) > 1.5:
            self.yawn_count += 1
            self.last_yawn_time = current_time
            print(f"检测到哈欠，当前哈欠次数: {self.yawn_count}")

    def check_evaluation_status(self):
        """检查评估状态，返回剩余时间或评估结果"""
        if not self.is_evaluating:
            return None

        current_time = time.time()
        elapsed = current_time - self.detection_start_time

        # 检查是否超时无人脸(1秒)
        if self.last_face_time and (current_time - self.last_face_time) > 1.0:
            self.cancel_evaluation()
            return None

        # 检查是否完成5秒评估
        if elapsed >= self.evaluation_duration:
            return self.finalize_evaluation()

        # 返回剩余时间
        return self.evaluation_duration - elapsed

    def finalize_evaluation(self):
        """完成评估并返回结果"""
        if not self.is_evaluating:
            return None

        # 计算各项指标
        perclos = self.eye_closed_frames / max(1, self.total_frames)
        blink_rate = self.blink_count / self.evaluation_duration  # 眨眼/秒
        yawn_count = self.yawn_count  # 5秒内哈欠次数

        # 使用评估器判断疲劳状态
        result = self.evaluator.evaluate(perclos, blink_rate, yawn_count, self.latest_window_features)
        self.evaluation_result = {
            "identity": self.current_identity,
            "fatigue": result['fatigue'],
            "score": result['score'],
            "perclos": result['perclos'],
            "blink_rate": result['blink_rate'],
            "yawn_count": result['yawn_count'],
            "longest_eye_closure": result["longest_eye_closure"],
            "max_yawn_duration": result["max_yawn_duration"],
            "window_features": result["window_features"],
            "weights": result['weights']
        }

        self.is_evaluating = False
        print(f"评估完成: {self.evaluation_result}")

        # 语音播报结果
        if self.tts_engine:
            try:
                status = "疲劳" if result['fatigue'] else "正常"
                # self.tts_engine.say(f"{self.current_identity}状态{status}")
                self.tts_engine.say(f"认证者状态{status}")
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"语音播报失败: {e}")

        return self.evaluation_result

    def cancel_evaluation(self):
        """取消评估"""
        self.is_evaluating = False
        self.evaluation_result = None
        print("检测不到人脸，已取消评估")


