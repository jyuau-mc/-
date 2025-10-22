from ultralytics import YOLO
import cv2
import numpy as np

class PersonCounter:
    def __init__(self, video_path, output_video_path='带统计的人流视频.mp4', model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.tracked_ids = set()
        self.tracker_config = {
            'track_thresh': 0.25,   
            'match_thresh': 0.8,    
            'track_buffer': 30      
        }
        # 初始化视频写入器
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

    def count_people(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  
            
            results = self.model.track(
                frame, 
                tracker='bytetrack.yaml',  
                conf=0.5,                  
                classes=0                  
            )

            if results[0].boxes.id is not None:
                current_ids = results[0].boxes.id.int().tolist()
                for track_id in current_ids:
                    self.tracked_ids.add(track_id)
            
            # 可视化：绘制跟踪框和人数
            annotated_frame = results[0].plot()
            cv2.putText(
                annotated_frame, 
                f"总人数: {len(self.tracked_ids)}", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2
            )
            # 将帧写入输出视频
            self.out.write(annotated_frame)

        print(f"视频中总人流量: {len(self.tracked_ids)}")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "videos/铜锣湾.mp4"  # 替换为实际视频路径
    counter = PersonCounter(video_path)  # 若需修改输出视频名，可传入output_video_path参数
    counter.count_people()