import cv2
import mediapipe as mp
import dlib
from deepface import DeepFace
import numpy as np
import json


class GestureAndEmotionRecognizer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.facial_expression_model = DeepFace.build_model("Emotion")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(0)

        with open('gestures.json') as f:
            self.gestures_dict = json.load(f)
        with open('special_gestures.json') as f:
            self.special_gestures_dict = json.load(f)
        with open('emotions.json') as f:
            self.emotion_dict = json.load(f)


    def calculate_distance(self, hand_landmarks, finger1, finger2):
        finger1_tip = hand_landmarks.landmark[finger1]
        finger2_tip = hand_landmarks.landmark[finger2]
        distance = np.sqrt((finger1_tip.x - finger2_tip.x) ** 2 + (finger1_tip.y - finger2_tip.y) ** 2)
        return distance


    def check_raised_fingers(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        thumb_raised = thumb_tip.y < thumb_mcp.y
        index_raised = index_tip.y < index_mcp.y
        middle_raised = middle_tip.y < middle_mcp.y
        ring_raised = ring_tip.y < ring_mcp.y
        pinky_raised = pinky_tip.y < pinky_mcp.y

        conditions = [thumb_raised, index_raised, middle_raised, ring_raised, pinky_raised]

        for gesture, gesture_conditions in self.gestures_dict.items():
            if conditions == gesture_conditions:
                return gesture

        for gesture, special_gesture in self.special_gestures_dict.items():
            if conditions == special_gesture["conditions"]:
                for finger_pair in special_gesture["finger_pairs"]:
                    if self.calculate_distance(hand_landmarks, finger_pair[0], finger_pair[1]) > special_gesture["max_distance"]:
                        break
                else:
                    return gesture

        return ""
    

    def check_emotions(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_image)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_image = image[y1:y2, x1:x2]
            emotion = DeepFace.analyze(face_image, actions=["emotion"], enforce_detection=False)

            try:
                dominant_emotion = emotion[0]["dominant_emotion"]

                if dominant_emotion in self.emotion_dict.keys():
                    cv2.putText(image, self.emotion_dict[dominant_emotion], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, dominant_emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            except:
                print('Emoção não encontrada.')

        return image


    def main(self):
        with self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self.check_emotions(image)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hand_results = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        text = self.check_raised_fingers(hand_landmarks)
                        hand_results.append(text)
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    for result in hand_results:
                        if result:
                            cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Totem Camera", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = GestureAndEmotionRecognizer()
    recognizer.main()
