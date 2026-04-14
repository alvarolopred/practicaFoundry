import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
import json

def prepare_dataset():
    print("🚀 Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("mubashirsidiki/student-academic-performance-500-students")
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        print("❌ No se encontró ningún archivo CSV.")
        return
    
    csv_path = os.path.join(path, files[0])
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset cargado. Columnas: {df.columns.tolist()}")
    
    conversations = []
    
    # Mensaje de sistema (Rol de Experto)
    system_content = "Eres un Asesor de Éxito Académico experto en análisis predictivo. Tu objetivo es evaluar el perfil de los estudiantes y determinar si tienen probabilidades de aprobar (Passed: Yes) o no (Passed: No) basándote en sus métricas."

    for _, row in df.iterrows():
        # Extraemos las características para la pregunta del usuario
        # Los nombres de columnas en este dataset suelen ser UpperCamelCase
        # Intentamos obtenerlos de forma robusta
        try:
            study_hours = row.get('StudyHoursPerWeek', row.get('study_hours_per_week', 'N/A'))
            attendance = row.get('AttendanceRate', row.get('attendance_rate', 'N/A'))
            prev_score = row.get('PreviousScore', row.get('previous_score', 'N/A'))
            extra = row.get('ExtracurricularActivities', row.get('extracurricular', 'N/A'))
            passed_val = str(row.get('Passed', row.get('passed', 'No'))).lower()
        except Exception:
            continue

        student_data = f"Horas de estudio: {study_hours}, Asistencia: {attendance}%, Puntuación previa: {prev_score}, Actividades extra: {extra}"
        
        # El resultado real del dataset
        resultado = "Aprobado (Yes)" if "yes" in passed_val else "No Aprobado (No)"
        
        conv = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Analiza mi perfil académico con estos datos: {student_data}. ¿Cuál es mi predicción de éxito?"},
                {"role": "assistant", "content": f"Tras analizar tus métricas, mi predicción basada en la evidencia es: {resultado}."}
            ]
        }
        conversations.append(conv)
    
    # División 80/20
    train_data, val_data = train_test_split(conversations, test_size=0.2, random_state=42)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(output_dir, "training_set.jsonl")
    val_path = os.path.join(output_dir, "validation_set.jsonl")
    
    # Guardar archivos
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"🎉 ¡Dataset generado correctamente con roles system, user y assistant!")
    print(f"📁 Entrenamiento: {len(train_data)} ejemplos")
    print(f"📁 Validación: {len(val_data)} ejemplos")

if __name__ == "__main__":
    prepare_dataset()
