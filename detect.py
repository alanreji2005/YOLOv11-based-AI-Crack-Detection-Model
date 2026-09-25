import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time
import threading

# 1. Background Camera Thread to instantly clear V4L2 buffers
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        # Constantly pull frames to prevent OpenCV buffer overflow
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.stopped = True
        self.cap.release()

# 2. Load the 100-epoch quantized TFLite model using all 4 CPU cores
interpreter = Interpreter(model_path="best_int8.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_size = 512

print("Starting background camera thread...")
stream = CameraStream(0).start()
time.sleep(1.0) # Allow camera to warm up

# 3. Initialize video writer to save the output over SSH
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('threaded_detection_output.mp4', fourcc, 10.0, (640, 480))

print("Recording threaded live camera feed with video saving. Press 'Ctrl+C' to stop.")

frame_count = 0
start_time = time.time()

try:
    while True:
        # 4. Read the absolute newest frame from the background thread
        ret, frame = stream.read()
        if not ret or frame is None:
            continue
            
        frame_count += 1
        original_h, original_w = frame.shape[:2]

        # 5. Preprocess the live frame (Letterboxing & RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = min(model_size / original_w, model_size / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        pad_w, pad_h = (model_size - new_w) / 2.0, (model_size - new_h) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img_normalized = img_padded.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_data = np.expand_dims(img_transposed, axis=0)

        # 6. Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # 7. Process the matrix with NMS
        predictions = np.transpose(output_data)
        boxes, confidences = [], []

        for row in predictions:
            conf = row[4]
            if conf > 0.35:  
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                if w <= 1.0 and h <= 1.0:
                    cx, cy, w, h = cx * model_size, cy * model_size, w * model_size, h * model_size
                    
                x_min = int(((cx - pad_w) - w / 2) / scale)
                y_min = int(((cy - pad_h) - h / 2) / scale)
                width, height = int(w / scale), int(h / scale)
                
                boxes.append([x_min, y_min, width, height])
                confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.35, nms_threshold=0.30)
        
        # 8. Draw boxes on the frame and print to terminal
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Crack {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Tracking Update [Inference {frame_count}]: Detected {len(indices)} crack(s)!")

        # 9. Write the annotated frame to the MP4 file
        out.write(frame)

except KeyboardInterrupt:
    elapsed_time = time.time() - start_time
    print(f"\nStopping inference. Processed and saved {frame_count} total AI inferences in {elapsed_time:.2f} seconds.")

finally:
    stream.stop()
    out.release()
    cv2.destroyAllWindows()
    print("Video successfully saved as threaded_detection_output.mp4")
