import cv2
import mediapipe as mp
import dlib
from deepface import DeepFace
import numpy as np

# Face
# Inicializando o detector de rosto do Dlib
detector = dlib.get_frontal_face_detector()
# Inicializando o modelo de reconhecimento de expressões faciais
facial_expression_model = DeepFace.build_model("Emotion")

# Gestos
# Inicializando os módulos do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Captura de vídeo
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            # Preparando a imagem
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image.flags.writeable = False
            # Processamento da imagem para detecção de rosto e reconhecimento de expressões faciais
            image = check_emotions(image)
            
            # Processando a imagem
            results = hands.process(image)

            # Desenhando as mãos
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_results = []  # Lista para armazenar os resultados das mãos detectadas
                for hand_landmarks in results.multi_hand_landmarks:
                    # Verificando as condições de dedos levantados
                    text = check_raised_fingers(hand_landmarks)
                    hand_results.append(text)  # Adicionando o resultado à lista
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                for result in hand_results:
                    if result:
                        cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Janela
            cv2.imshow("Totem Camera", image)
            # 'q' para fechar janela
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# Função para checar se os dedos estão levantados / Retorna string para label
def calculate_distance(hand_landmarks, finger1, finger2):
    finger1_tip = hand_landmarks.landmark[finger1]
    finger2_tip = hand_landmarks.landmark[finger2]

    distance = np.sqrt((finger1_tip.x - finger2_tip.x)**2 + (finger1_tip.y - finger2_tip.y)**2)

    return distance

def check_raised_fingers(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    thumb_raised = thumb_tip.y < thumb_mcp.y
    index_raised = index_tip.y < index_mcp.y
    middle_raised = middle_tip.y < middle_mcp.y
    ring_raised = ring_tip.y < ring_mcp.y
    pinky_raised = pinky_tip.y < pinky_mcp.y

    conditions = [thumb_raised, index_raised, middle_raised, ring_raised, pinky_raised]

    gestures_dict = {
        "Hoje nao e o dia do Rock": [True, True, False, False, True],
        "Paz e Amor": [False, True, True, False, False],
        "Tudo Legal": [True, False, False, False, False]
    }

    special_gestures_dict = {
        "Também te amo": {
            "conditions": [True, True, True, False, False],
            "finger_pairs": [(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP)],
            "max_distance": 0.2
        }
        # Adicione outros gestos especiais aqui...
    }

    for gesture, gesture_conditions in gestures_dict.items():
        if conditions == gesture_conditions:
            return gesture

    for gesture, special_gesture in special_gestures_dict.items():
        if conditions == special_gesture["conditions"]:
            for finger_pair in special_gesture["finger_pairs"]:
                if calculate_distance(hand_landmarks, finger_pair[0], finger_pair[1]) > special_gesture["max_distance"]:
                    break
            else:
                return gesture

    return ""


# Função para detectar emoções
def check_emotions(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Cortando o rosto para análise de expressões faciais
        face_image = image[y1:y2, x1:x2]

        # Reconhecendo a expressão facial
        emotion = DeepFace.analyze(face_image, actions=["emotion"], enforce_detection=False)
        #print(emotion)
        try:
            dominant_emotion = emotion[0]["dominant_emotion"]
            # Colocando a expressão facial no quadro
            emotion_dict = {"happy":"Estou vendo que voce esta feliz", "neutral":"", "sad":"", "angry":"", "fear":"", "surprise":""}
            if dominant_emotion in emotion_dict.keys():
                cv2.putText(image, emotion_dict[dominant_emotion], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, dominant_emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            print('Emoção não encontrada.')
    return image

if __name__ == "__main__":
    main()
