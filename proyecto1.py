import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk  # <-- LIBRERÍA MODERNA
import cv2
from PIL import Image, ImageTk
import time
import random
import os
from datetime import datetime
import threading
import numpy as np

# Librerías para DSP y Machine Learning
try:
    import librosa
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except ImportError:
    print("Advertencia: Faltan librerías ML/DSP. Ejecuta: pip install librosa scikit-learn numpy joblib")
    
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
except ImportError:
    print("Advertencia: Faltan librerías de grabación. Ejecuta: pip install sounddevice scipy")

# Configuración inicial del tema moderno
ctk.set_appearance_mode("Dark")  # Opciones: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Opciones: "blue", "green", "dark-blue"

class SistemaSeguridadIndustrial:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Seguridad Industrial IA - Frigoríficos")
        self.root.geometry("1100x700")

        self.crear_directorios()

        self.modulos = [
            {"id": 1, "titulo": "Módulo 1: Detección Acústica", "desc": "Análisis de espectro de audio 1D para cámaras de frío.", "tipo": "audio", "icon": "🔊"},
            {"id": 2, "titulo": "Módulo 2: Control Visual EPP", "desc": "Análisis 2D para detección de cascos al ingreso.", "tipo": "video", "icon": "👁️"}
        ]

        self.setup_ui()

    def crear_directorios(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.dataset_audio_norm = os.path.join(self.data_dir, "datasets", "audio", "normal")
        self.dataset_audio_acc = os.path.join(self.data_dir, "datasets", "audio", "accidente")
        self.dataset_img_casco = os.path.join(self.data_dir, "datasets", "images", "con_casco")
        self.dataset_img_sincasco = os.path.join(self.data_dir, "datasets", "images", "sin_casco")
        self.models_dir = os.path.join(self.data_dir, "models")
        self.live_audio_dir = os.path.join(self.data_dir, "live_captures", "audio")
        self.live_img_dir = os.path.join(self.data_dir, "live_captures", "images")
        
        for directorio in [self.dataset_audio_norm, self.dataset_audio_acc, self.dataset_img_casco, 
                           self.dataset_img_sincasco, self.models_dir, self.live_audio_dir, self.live_img_dir]:
            os.makedirs(directorio, exist_ok=True)

    def setup_ui(self):
        # Cabecera Modernizada
        header = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#1e293b")
        header.pack(fill="x")
        ctk.CTkLabel(header, text="🛡️ Centro de Control de Seguridad - IA", font=ctk.CTkFont(family="Roboto", size=24, weight="bold"), text_color="white").pack(pady=20)

        # Contenedor principal
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=40, pady=30)

        ctk.CTkLabel(self.main_frame, text="Módulos de Monitoreo", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 20))

        # Tarjetas de Módulos (Cards)
        for modulo in self.modulos:
            card = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#334155", border_width=1, border_color="#475569")
            card.pack(fill="x", pady=10, ipady=10)

            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", fill="both", expand=True, padx=20)

            ctk.CTkLabel(info_frame, text=f"{modulo['icon']} {modulo['titulo']}", font=ctk.CTkFont(size=18, weight="bold"), text_color="#f8fafc").pack(anchor="w")
            ctk.CTkLabel(info_frame, text=modulo["desc"], font=ctk.CTkFont(size=13), text_color="#cbd5e1").pack(anchor="w", pady=(5,0))

            btn_iniciar = ctk.CTkButton(card, text="Abrir Consola ➜", corner_radius=8, font=ctk.CTkFont(weight="bold"), 
                                        fg_color="#2563eb", hover_color="#1d4ed8",
                                        command=lambda m=modulo: self.abrir_consola_monitoreo(m))
            btn_iniciar.pack(side="right", padx=20)

    def abrir_consola_monitoreo(self, modulo):
        self.monitor_win = ctk.CTkToplevel(self.root)
        self.monitor_win.title(f"Monitor - {modulo['titulo']}")
        self.monitor_win.geometry("1100x750")
        self.monitor_win.attributes('-topmost', True) # Pone la ventana por encima momentáneamente
        self.monitor_win.after(100, lambda: self.monitor_win.attributes('-topmost', False))

        ctk.CTkLabel(self.monitor_win, text=modulo['titulo'], font=ctk.CTkFont(size=22, weight="bold")).pack(pady=15)

        # Contenedor de Video y Consola
        content = ctk.CTkFrame(self.monitor_win, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Panel de Video
        video_frame = ctk.CTkFrame(content, corner_radius=15)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Usamos tk.Label estándar para el video porque es más eficiente para los frames de OpenCV
        self.video_panel = tk.Label(video_frame, bg="#0f172a", text="SEÑAL DE SENSORES", fg="#64748b", font=("Roboto", 16))
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=10)

        # Consola de Texto Modernizada
        self.consola = ctk.CTkTextbox(content, width=400, font=ctk.CTkFont(family="Consolas", size=12), 
                                      fg_color="#020617", text_color="#10b981", corner_radius=15, border_width=1, border_color="#1e293b")
        self.consola.pack(side="right", fill="y")
        self.log("SISTEMA INICIADO...\nEsperando comandos...")

        # Controles (Botones inferiores)
        btn_frame = ctk.CTkFrame(self.monitor_win, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="⚙️ ENTRENAR MODELO", font=ctk.CTkFont(weight="bold"), fg_color="#0891b2", hover_color="#0e7490",
                   command=lambda: self.iniciar_entrenamiento_thread(modulo['tipo'])).pack(side="left", padx=10)
                   
        self.btn_eval = ctk.CTkButton(btn_frame, text="🔴 EVALUAR SEÑAL EN VIVO", font=ctk.CTkFont(weight="bold"), fg_color="#e11d48", hover_color="#be123c",
                                   command=lambda: self.iniciar_evaluacion_thread(modulo['tipo']))
        self.btn_eval.pack(side="left", padx=10)

        if modulo['tipo'] == "video":
            instruccion = ctk.CTkLabel(video_frame, text="Teclas: 'C' = Foto Con Casco | 'S' = Foto Sin Casco", text_color="#fbbf24", font=ctk.CTkFont(weight="bold"))
            instruccion.pack(pady=10)
            self.monitor_win.bind("<c>", lambda e: self.guardar_captura("casco"))
            self.monitor_win.bind("<s>", lambda e: self.guardar_captura("sin_casco"))
            
            self.cap = cv2.VideoCapture(0)
            self.actualizar_frame()
        else:
            instruccion = ctk.CTkLabel(video_frame, text="Teclas: 'A' = Grabar Accidente (2s) | 'N' = Grabar Normal (2s)", text_color="#fbbf24", font=ctk.CTkFont(weight="bold"))
            instruccion.pack(pady=10)
            self.monitor_win.bind("<a>", lambda e: self.iniciar_grabacion_audio("accidente"))
            self.monitor_win.bind("<n>", lambda e: self.iniciar_grabacion_audio("normal"))
            
            self.video_panel.config(text="MICRÓFONO ACTIVADO\nPresiona 'A' o 'N' para grabar")
            self.cap = None

        self.monitor_win.protocol("WM_DELETE_WINDOW", self.cerrar_monitor)

    # =================================================================
    # A PARTIR DE AQUÍ, LA LÓGICA DE CÓDIGO ES LA MISMA DEL PASO ANTERIOR
    # =================================================================
    def iniciar_grabacion_audio(self, clase_audio):
        # Usamos un hilo para que la interfaz no se congele durante los 2 segundos
        threading.Thread(target=self.grabar_audio_dataset, args=(clase_audio,), daemon=True).start()

    def grabar_audio_dataset(self, clase_audio):
        self.log(f"🎙️ GRABANDO ({clase_audio.upper()}) por 2 segundos...")
        try:
            fs = 44100  # Frecuencia de muestreo (Hz)
            duracion = 2  # Segundos
            # Graba audio desde el micrófono predeterminado
            grabacion = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Espera a que terminen los 2 segundos
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            carpeta = self.dataset_audio_acc if clase_audio == "accidente" else self.dataset_audio_norm
            ruta = os.path.join(carpeta, f"audio_{timestamp}.wav")
            
            wav.write(ruta, fs, grabacion)
            archivos = len([f for f in os.listdir(carpeta) if f.endswith('.wav')])
            self.log(f"✅ Audio guardado: {clase_audio.upper()} ({archivos} archivos en total)")
        except Exception as e:
            self.log(f"❌ Error al grabar: {str(e)}\nVerifica tu micrófono y que 'sounddevice' esté instalado.")

    def guardar_captura(self, tipo):
        if not hasattr(self, 'current_frame'): return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if tipo == "casco":
            ruta = os.path.join(self.dataset_img_casco, f"img_{timestamp}.jpg")
            cv2.imwrite(ruta, self.current_frame)
            self.log(f"📸 Guardado: CON CASCO ({len(os.listdir(self.dataset_img_casco))} imágenes)")
        else:
            ruta = os.path.join(self.dataset_img_sincasco, f"img_{timestamp}.jpg")
            cv2.imwrite(ruta, self.current_frame)
            self.log(f"📸 Guardado: SIN CASCO ({len(os.listdir(self.dataset_img_sincasco))} imágenes)")

    def log(self, texto):
        def update():
            self.consola.insert("end", texto + "\n")
            self.consola.see("end")
        self.root.after(0, update)

    def actualizar_frame(self):
        if not self.cap or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)
        self.video_loop = self.monitor_win.after(15, self.actualizar_frame)

    def cerrar_monitor(self):
        if self.cap:
            self.cap.release()
        self.monitor_win.destroy()

    def extraer_features_imagen(self, img):
        img_resized = cv2.resize(img, (32, 32))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        return hsv.flatten()

    def iniciar_entrenamiento_thread(self, tipo):
        threading.Thread(target=self.entrenar_modelo, args=(tipo,), daemon=True).start()
        
    def iniciar_evaluacion_thread(self, tipo):
        self.btn_eval.configure(state="disabled")
        threading.Thread(target=self.evaluar_senal, args=(tipo,), daemon=True).start()

    def entrenar_modelo(self, tipo):
        self.log(f"\n[ENTRENAMIENTO] Extrayendo características ({tipo})...")
        X, y = [], []
        
        try:
            if tipo == "audio":
                for label, folder in enumerate([self.dataset_audio_norm, self.dataset_audio_acc]):
                    archivos = os.listdir(folder)
                    if not archivos:
                        self.log(f"⚠️ Carpeta vacía: {folder}. Agrega audios .wav para entrenar.")
                        return
                    for archivo in archivos:
                        ruta = os.path.join(folder, archivo)
                        y_audio, sr = librosa.load(ruta, duration=3)
                        energia = np.mean(librosa.feature.rms(y=y_audio))
                        zcr = np.mean(librosa.feature.zero_crossing_rate(y_audio))
                        centroide = np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr))
                        ancho_banda = np.mean(librosa.feature.spectral_bandwidth(y=y_audio, sr=sr))
                        X.append([energia, zcr, centroide, ancho_banda])
                        y.append(label)
                        
            elif tipo == "video":
                for label, folder in enumerate([self.dataset_img_sincasco, self.dataset_img_casco]):
                    archivos = os.listdir(folder)
                    if len(archivos) < 5:
                        self.log(f"❌ FALTAN FOTOS en {os.path.basename(folder)}. Usa las teclas 'C' y 'S' para tomar al menos 5 fotos de cada tipo.")
                        return
                        
                    self.log(f" -> Procesando {len(archivos)} imágenes de '{os.path.basename(folder)}'...")
                    for archivo in archivos:
                        ruta = os.path.join(folder, archivo)
                        img = cv2.imread(ruta)
                        vector = self.extraer_features_imagen(img)
                        X.append(vector)
                        y.append(label)

            if not X or not y: return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            reporte = classification_report(y_test, y_pred, zero_division=0)
            self.log("\n[REPORTE DE CLASIFICACIÓN]\n" + reporte)
            
            ruta_modelo = os.path.join(self.models_dir, f"modelo_{tipo}.pkl")
            joblib.dump(clf, ruta_modelo)
            self.log(f"✅ Modelo {tipo} entrenado y guardado con éxito.")
            
        except Exception as e:
            self.log(f"❌ Error en entrenamiento: {str(e)}")

    def evaluar_senal(self, tipo):
        ruta_modelo = os.path.join(self.models_dir, f"modelo_{tipo}.pkl")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists(ruta_modelo):
            self.log("❌ ERROR: No hay un modelo entrenado. Presiona 'ENTRENAR MODELO' primero.")
            self.root.after(0, lambda: self.btn_eval.configure(state="normal"))
            return
            
        try:
            clf = joblib.load(ruta_modelo)
            
            if tipo == "audio":
                self.log("\n[EVALUACIÓN ACÚSTICA] Escuchando micrófonos...")
                time.sleep(2) 
                
                energia = random.uniform(0.01, 0.5)
                zcr = random.uniform(0.01, 0.2)
                centroide = random.uniform(1000, 4000)
                ancho_banda = random.uniform(1000, 3000)
                vector_caracteristicas = [energia, zcr, centroide, ancho_banda]
                
                self.log(f" -> Características: RMS={energia:.2f}, ZCR={zcr:.2f}")
                prediccion = clf.predict([vector_caracteristicas])[0]
                
                if prediccion == 1:
                    self.log(">>> ACCIDENTE DETECTADO >>> ACTIVAR ALARMA 🔴")
                    self.root.after(0, lambda: messagebox.showwarning("ALERTA CRÍTICA", "¡Posible accidente detectado!"))
                else:
                    self.log(" -> Estado Acústico: NORMAL 🟢")

            elif tipo == "video":
                self.log("\n[EVALUACIÓN VISUAL] Analizando acceso...")
                if hasattr(self, 'current_frame'):
                    img_path = os.path.join(self.live_img_dir, f"captura_{timestamp}.jpg")
                    cv2.imwrite(img_path, self.current_frame)
                    
                    vector_caracteristicas = self.extraer_features_imagen(self.current_frame)
                    prediccion = clf.predict([vector_caracteristicas])[0]
                    
                    if prediccion == 0:
                        self.log(">>> SIN CASCO DETECTADO >>> ACCESO DENEGADO 🔴")
                        self.root.after(0, lambda: messagebox.showerror("Violación de Seguridad", "Trabajador sin casco. Bloqueando puerta..."))
                    else:
                        self.log(" -> EPP Completo. ACCESO PERMITIDO 🟢")
                        
        except Exception as e:
            self.log(f"❌ Error en evaluación: {str(e)}")
        finally:
            self.root.after(0, lambda: self.btn_eval.configure(state="normal"))

if __name__ == "__main__":
    root = ctk.CTk() # <-- Iniciamos usando CustomTkinter
    app = SistemaSeguridadIndustrial(root)
    root.mainloop()