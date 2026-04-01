import cv2
import mediapipe as mp
import numpy as np

class CameraEvaluator:
    def __init__(self):
        import os
        self.model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=2)
            # Create detector
            self.detector_creator = lambda: vision.FaceLandmarker.create_from_options(options)
            self.mp_image = mp.Image
            self.mp_image_format = mp.ImageFormat

        except Exception as e:
            print(f"Warning: mediapipe FaceLandmarker could not be loaded. Error: {e}")
            if not os.path.exists(self.model_path):
                print(f"Make sure '{self.model_path}' exists.")
            self.detector_creator = None
        # Landmark Indices
        self.LEFT_EYE = [362, 385, 386, 263, 374, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.IRIS_L = 468
        self.IRIS_R = 473
        self.NOSE_TIP = 1
        self.MOUTH_OUTER = [61, 291, 0, 17] # Corners (L, R) and Lip boundaries
        self.MOUTH_INNER = [13, 14, 78, 308] # Inner lips for MAR
        
        # 3D Model Points for Pose Estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

    def _get_pose(self, landmarks, img_w, img_h):
        """Estimates 3D Head Pose (Yaw, Pitch, Roll)."""
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
            (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye corner
            (landmarks[263].x * img_w, landmarks[263].y * img_h), # Right eye corner
            (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left mouth corner
            (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth corner
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Adjust for coordinate system
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        return pitch, yaw, roll

    def _calculate_mar(self, landmarks):
        """Calculates Mouth Aspect Ratio (MAR)."""
        try:
            # Vertical distances
            v1 = np.linalg.norm(np.array([landmarks[13].x, landmarks[13].y]) - 
                               np.array([landmarks[14].x, landmarks[14].y]))
            # Horizontal distance
            h = np.linalg.norm(np.array([landmarks[78].x, landmarks[78].y]) - 
                              np.array([landmarks[308].x, landmarks[308].y]))
            return v1 / h if h > 0 else 0
        except:
            return 0

    def _calculate_smile_ratio(self, landmarks):
        try:
            h_dist = np.linalg.norm(np.array([landmarks[61].x, landmarks[61].y]) - 
                                   np.array([landmarks[291].x, landmarks[291].y]))
            eye_dist = np.linalg.norm(np.array([landmarks[33].x, landmarks[33].y]) - 
                                     np.array([landmarks[263].x, landmarks[263].y]))
            return h_dist / eye_dist
        except:
            return 0.5

    def _calculate_ear(self, landmarks, eye_indices):
        try:
            v1 = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                                np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                                np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                               np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        except:
            return 0.3

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frames, face_detected, eye_contact, blinks, smiles = 0, 0, 0, 0, 0
        mars, head_yaw, head_pitch, multi_face_count = [], [], [], 0
        blink_active = False

        if not self.detector_creator:
            return {"camera_score": 0, "error": "Mediapipe FaceLandmarker not installed/loaded."}

        # Create the detector and ensure we close it to prevent Python 3.13 GC crashes
        detector = self.detector_creator()
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frames += 1
                if frames % 8 != 0:
                    continue # Process every 8th frame for speed
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = self.mp_image(image_format=self.mp_image_format.SRGB, data=rgb_frame)
                
                # Predict
                results = detector.detect(mp_img)
                
                if results.face_landmarks:
                    face_detected += 1
                    if len(results.face_landmarks) > 1:
                        multi_face_count += 1
                        
                    mesh = results.face_landmarks[0]
                    
                    # 1. Pose & Eye Contact
                    pitch, yaw, roll = self._get_pose(mesh, img_w, img_h)
                    head_yaw.append(abs(yaw))
                    head_pitch.append(pitch)
                    
                    if face_detected % 50 == 0:
                        print(f"DEBUG: Frame {frames} - Yaw: {yaw:.2f}, Pitch: {pitch:.2f}")

                    # 2. Iris Analysis (Robust Centering)
                    try:
                        # Use min/max to be robust against mirrored frames or coordinate flips
                        x_362, x_263 = mesh[362].x, mesh[263].x
                        l_min, l_max = min(x_362, x_263), max(x_362, x_263)
                        
                        x_33, x_133 = mesh[33].x, mesh[133].x
                        r_min, r_max = min(x_33, x_133), max(x_33, x_133)

                        if (l_max - l_min) > 0.005 and (r_max - r_min) > 0.005:
                            # FIX: 478-landmark model mapping:
                            # Right Eye in image: corners (362, 263), Iris (473)
                            # Left Eye in image: corners (33, 133), Iris (468)
                            rel_l = (mesh[self.IRIS_R].x - l_min) / (l_max - l_min)
                            rel_r = (mesh[self.IRIS_L].x - r_min) / (r_max - r_min)
                            
                            # STRICTOR: Narrowed range 0.35 - 0.65 for more accurate gaze
                            # Also require BOTH eyes to be centered if possible (or very solid single eye)
                            if 0.35 < rel_l < 0.65 and 0.35 < rel_r < 0.65: 
                                eye_contact += 1
                            
                            if face_detected % 50 == 0:
                                print(f"DEBUG: Iris Centering - Rel_L: {rel_l:.3f}, Rel_R: {rel_r:.3f}")
                        else:
                            if face_detected % 50 == 0:
                                print(f"DEBUG: Eye width too small - L: {l_max-l_min:.4f}, R: {r_max-r_min:.4f}")
                    except IndexError:
                        if face_detected % 50 == 0:
                            print("DEBUG: IndexError during Iris analysis")
                        pass

                    # 3. Speech Activity (MAR)
                    mar = float(self._calculate_mar(mesh))
                    mars.append(mar)
                    
                    # 4. Blink & Smile
                    ear = float((self._calculate_ear(mesh, self.LEFT_EYE) + self._calculate_ear(mesh, self.RIGHT_EYE)) / 2.0)
                    if ear < 0.22:
                        if not blink_active:
                            blinks += 1
                            blink_active = True
                    else:
                        blink_active = False
                    
                    smile_ratio = float(self._calculate_smile_ratio(mesh))
                    # Lowered threshold from 1.1 to 1.0 to detect subtle smiles
                    if smile_ratio > 1.0:
                        smiles += 1
        finally:
            detector.close()

        cap.release()
        if frames == 0: return {"camera_score": 0, "error": "No frames"}

        # Metrics Breakdown
        duration = frames / fps
        processed_frames = max(1, frames // 4)
        face_visibility = (face_detected / processed_frames) * 100
        
        # FIX: Calculate eye contact relative to total session frames, not just detected faces
        # This prevents 100% eye contact if only 1 frame was detected.
        eye_pct = (eye_contact / processed_frames) * 100
        
        avg_yaw = np.mean(head_yaw) if head_yaw else 0
        moving_mouth_pct = (len([m for m in mars if m > 0.15]) / processed_frames) * 100
        multi_face_pct = (multi_face_count / processed_frames) * 100

        # Logic adjustment: High Accuracy Mode
        # If face visibility is low (<20%) or detections are too few (<15 frames), zero out metrics
        # This prevents "noise" from hands/objects from getting behavioral credit.
        if face_visibility < 20 or face_detected < 15:
            eye_pct = 0.0
            moving_mouth_pct = 0.0
            smiles = 0
            face_visibility = face_visibility if face_detected > 5 else 0.0 # Clear visibility if near zero

        # Scoring Logic (10-point scale)
        # Eye Contact: Up to 4 points (Need >70% to get full 4 points)
        eye_score = min(4.0, (eye_pct / 70.0) * 4.0)
        
        # Face Visibility: Up to 3 points (Need >80% for full points)
        vis_score = min(3.0, (face_visibility / 80.0) * 3.0)
        
        # Pose Stability (Low Yaw): Up to 2 points (Ideal yaw < 15 degrees)
        pose_score = max(0.0, 2.0 - (max(0, avg_yaw - 15) / 10.0))
        
        # Expression/Engagement: Up to 1 point (smiles or slight mouth movement)
        expr_score = min(1.0, (smiles / max(1, face_detected) * 1.0) + (moving_mouth_pct / 50.0))
        
        score = eye_score + vis_score + pose_score + expr_score
        
        # Harsh Penalties
        if multi_face_pct > 10: score -= 5.0 # Severe cheating penalty
        if moving_mouth_pct < 5 and face_visibility > 50: score -= 2.0 # Silent/Not speaking?
        if avg_yaw > 25: score -= 2.0 # Looking away
        
        # New: Visibility Penalties
        if face_visibility < 40: score -= 3.0
        if face_visibility < 15: score = 0.0 # Effectively no face present
        
        final_score = round(max(0, min(10, score)), 1)
        
        # Dynamic Feedback
        feedback = f"Maintained eye contact for {eye_pct:.0f}% of session."
        if moving_mouth_pct < 10: feedback += " We detected very little mouth movement; ensure you are speaking clearly."
        if avg_yaw > 20: feedback += " You frequently looked away from the camera. Try to stay centered."
        if multi_face_pct > 5: feedback += " WARNING: Multiple faces detected. Ensure you are alone during evaluations."

        return {
            "camera_score": float(final_score),
            "metrics": {
                "eye_contact": float(round(eye_pct, 1)),
                "smile_pct": float(round((smiles / processed_frames) * 100, 1)),
                "speech_activity": float(round(moving_mouth_pct, 1)),
                "head_yaw_avg": float(round(avg_yaw, 1)),
                "visibility": float(round(face_visibility, 1)),
                "multi_face_violation": bool(multi_face_pct > 10)
            },
            "camera_feedback": feedback
        }

def analyze_camera(video_path: str) -> dict:
    """Helper function to instantiate the evaluator and run analysis, for compatibility."""
    evaluator = CameraEvaluator()
    return evaluator.analyze_video(video_path)

# Usage Example:
# evaluator = CameraEvaluator()
# result = evaluator.analyze_video("user_gd_video.mp4")
# print(result)
