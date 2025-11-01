import cv2 as cv
import numpy as np

# --- 1. Definición de Colores ---
# Guardamos los colores en un diccionario para manejarlos fácilmente.
# Cada color tiene sus rangos HSV (lower, upper) y el color BGR para dibujar.
color_ranges = {
    'amarillo': {
        'lower': np.array([15, 150, 20]),
        'upper': np.array([35, 255, 255]),
        'bgr': (0, 255, 255)  # Amarillo en BGR
    },
    'azul': {
        'lower': np.array([90, 100, 20]),
        'upper': np.array([130, 255, 255]),
        'bgr': (255, 0, 0)  # Azul en BGR
    },
    'rojo': {
        # El rojo necesita dos rangos en HSV
        'lower1': np.array([0, 150, 20]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([170, 150, 20]),
        'upper2': np.array([180, 255, 255]),
        'bgr': (0, 0, 255)  # Rojo en BGR
    }
}

# --- 2. Kernel para Reducción de Ruido ---
# Creamos un kernel (matriz) para las operaciones morfológicas
# Un kernel 5x5 es un buen punto de partida.
kernel = np.ones((5, 5), np.uint8)

# --- 3. Captura de Video ---
webcam = cv.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convertimos la imagen a HSV
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # --- 4. Bucle de Detección para CADA color ---
    for color_name, limits in color_ranges.items():
        
        # --- 4a. Crear la Máscara ---
        if color_name == 'rojo':
            # Combinamos las dos máscaras para el rojo
            mask1 = cv.inRange(img_hsv, limits['lower1'], limits['upper1'])
            mask2 = cv.inRange(img_hsv, limits['lower2'], limits['upper2'])
            mask = cv.add(mask1, mask2)
        else:
            # Máscara normal para azul y amarillo
            mask = cv.inRange(img_hsv, limits['lower'], limits['upper'])

        # --- 4b. Reducción de Ruido ---
        # Aplicamos "Opening" (primero erosiona y luego dilata)
        # Esto elimina los pequeños puntos blancos (ruido) de la máscara.
        mask_denoised = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        # Opcional: puedes aplicar "Closing" para rellenar huecos negros dentro
        # de los objetos blancos.
        mask_denoised = cv.morphologyEx(mask_denoised, cv.MORPH_CLOSE, kernel)

        # --- 4c. Encontrar Contornos ---
        contours, hierarchy = cv.findContours(mask_denoised, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            for contour in contours:
                # Aumenté el área mínima a 500 para filtrar más ruido
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    
                    # Dibujar el rectángulo (usando el color BGR definido)
                    cv.rectangle(frame, (x, y), (x + w, y + h), limits['bgr'], 3)
                    
                    # --- 4d. Poner el "Pop-up" (Texto) ---
                    # Escribimos el nombre del color justo arriba del recuadro
                    cv.putText(frame, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, limits['bgr'], 2)

    # --- 5. Mostrar el Frame ---
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):  # Presiona 'd' para detener
        break

webcam.release()
cv.destroyAllWindows()