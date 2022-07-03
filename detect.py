from sahi.model import Yolov6DetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

detection_model = Yolov6DetectionModel(
    model_path='yolov6s.pt',
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
    image_size=640,
)

#result = get_prediction("demo/demo_data/highway.jpg", detection_model)

result = get_sliced_prediction(
    'demo/demo_data/highway.jpg',
    detection_model,
    slice_height = 1280,
    slice_width = 1280,
    overlap_height_ratio = 0.6,
    overlap_width_ratio = 0.6,
)

result.export_visuals(export_dir="demo_data/")
