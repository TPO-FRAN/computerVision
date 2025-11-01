import cv2 as cv
import numpy as np

# --- 1. Definimos los colores a rastrear ---
# Cada color tiene sus límites HSV y un color BGR para dibujar el rectángulo
colors_to_track = {
    "rojo_1": {
        "lower": np.array([0, 150, 20]),
        "upper": np.array([10, 255, 255]),
        "draw_color": (0, 0, 255)  # BGR: Rojo
    },
    "rojo_2": {
        "lower": np.array([170, 150, 20]), 
        "upper": np.array([180, 255, 255]),
        "draw_color": (0, 0, 255)  # BGR: Rojo
    },
    # Azul y Amarillo solo necesitan un rango
    "azul": {
        "lower": np.array([90, 100, 20]),
        "upper": np.array([130, 255, 255]),
        "draw_color": (255, 0, 0)    # BGR: Azul
    },
    "amarillo": {
        "lower": np.array([15, 150, 20]),
        "upper": np.array([35, 255, 255]),
        "draw_color": (0, 255, 255)  # BGR: Amarillo
    }
}

webcam = cv.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convertir a HSV (solo se hace una vez)
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # --- 2. Iteramos sobre cada color ---
    for color_name, data in colors_to_track.items():
        
        # 3. Creamos una máscara para el color actual
        mask = cv.inRange(img_hsv, data["lower"], data["upper"])

        # 4. Encontramos contornos para la máscara actual
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 5. Dibujamos los rectángulos
        if len(contours) != 0:
            for contour in contours:
                if cv.contourArea(contour) > 300:
                    x, y, w, h = cv.boundingRect(contour)
                    
                    # Usamos el color de dibujo específico de 'data'
                    cv.rectangle(frame, (x, y), (x + w, y + h), data["draw_color"], 3)
                    
                    # (Opcional) Escribimos el nombre del color
                    cv.putText(frame, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, data["draw_color"], 2)
    

    cv.imshow('frame', frame)
    # (Opcional) Si quieres ver las máscaras individuales, puedes agregarlas
    # cv.imshow('mask_amarillo', cv.inRange(img_hsv, colors_to_track["amarillo"]["lower"], colors_to_track["amarillo"]["upper"]))
    
    if cv.waitKey(20) & 0xFF == ord('s'):
        break

webcam.release()
cv.destroyAllWindows()