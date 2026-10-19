import cv2

# 0 es el índice de la cámara integrada o la primera conectada por USB
cap = cv2.VideoCapture(0)

while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    # Mostrar el video en una ventana
    cv2.imshow('Mi Proyecto - Cámara', frame)

    # Detener con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
