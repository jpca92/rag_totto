README — Asistente GenAI con RAG, Embeddings, FAISS y Datos Estructurados
Descripción del Proyecto

Este proyecto implementa un Asistente GenAI corporativo que combina:

RAG (Retrieval Augmented Generation) para responder preguntas basadas en documentos internos (FAQ, políticas, términos).

Modelos de Embedding (BERT MiniLM) para indexar texto.

FAISS como vector store para búsquedas semánticas eficientes.

LLM (OpenAI GPT-5-nano) para generar respuestas naturales y contextuales.

Datos estructurados (CSV) para consultar información de inventario y atributos de productos mediante SKU.

El asistente detecta automáticamente si la consulta del usuario contiene un SKU válido y responde primero con información estructurada (stock, impermeabilidad) y luego integra este contexto con información de documentos usando RAG + LLM para producir una respuesta única, clara y profesional.

Características principales
1. Detección automática de SKU

Cualquier palabra que comience por "SKU" se interpreta como un SKU válido.

Permite consultas como:

"SKU100732 se puede devolver?"

"hay disponibilidad del sku100200?"

"el SKU100900 es impermeable?"

2. Respuestas híbridas SKU + RAG

Si la consulta contiene un SKU, el asistente:

Obtiene stock desde inventory.csv.

Obtiene atributos del producto desde products.csv.

Recupera documentos relevantes mediante FAISS.

Genera una sola respuesta integrada con toda la información.

3. RAG completo para FAQs

Si la consulta no contiene SKU, se usa:

FAISS para recuperar chunks relevantes

BERT MiniLM para embeddings

GPT-4o-mini para responder con citas

4. Modo Debug opcional

En el panel lateral puedes activar un modo debug que muestra:

Chunks recuperados

Scores de FAISS

Prompt enviado al LLM

Ideal para demos, pruebas técnicas y validación del RAG.

5. Aceleración por GPU (opcional)

Si hay una GPU disponible (torch.cuda.is_available()), el modelo de embeddings correrá sobre CUDA automáticamente.

📂 Estructura del Proyecto
project/
│
├─ app.py                         # Aplicación Streamlit principal
├─ README.md                      # Este archivo
│
├─ data/
│   ├─ products.csv               # Catálogo de productos
│   ├─ inventory.csv              # Inventario por SKU
│   ├─ faq.md                     # Preguntas frecuentes
│   ├─ politica_devoluciones.txt
│   ├─ politica_garantias.txt
│   ├─ terminos_servicio.txt
│   └─ otros documentos .txt/.md
│
└─ requirements.txt                # Dependencias del proyecto

- Requisitos

Asegúrate de tener instalado:

Python 3.12

pip

Entorno virtual es opcional pero recomendado.

- Instalación
1. Crear entorno virtual
python -m venv venv

2. Activarlo

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate

3. Instalar dependencias
pip install -r requirements.txt

4. Configurar clave OpenAI
setx OPENAI_API_KEY "TU_API_KEY"


Cierra y abre la consola después de usar setx.

- Ejecutar la aplicación

En la raíz del proyecto:

streamlit run app.py


La interfaz estará disponible en:

http://localhost:8501

- Ejemplos de uso
Consulta con SKU
el SKU100732 se puede devolver?


Respuesta integrada:

Stock disponible

Impermeabilidad

Política de devoluciones (RAG)

Cita de fuente

Consulta sin SKU
¿Cómo tramito una devolución?


Respuesta:

Recuperada desde documentos FAQ

Generada por RAG + LLM

Modo Debug

Activa el interruptor en el sidebar para ver:

Fragmentos recuperados

Scores de similitud

Prompt completo enviado al LLM

- Tecnologías utilizadas
Componente	Tecnología
Motor GenAI	OpenAI GPT-5-nano
Embeddings	Sentence-BERT MiniLM
Vector Store	FAISS FlatL2
UI	Streamlit
Procesamiento CSV	Pandas
Aceleración	PyTorch GPU opcional

- Notas Técnicas

El sistema evita que RAG interfiera cuando la consulta requiere información estructurada, lo cual es una práctica estándar en asistentes corporativos.

Los embeddings se cachean para optimizar tiempos de carga.

Los documentos se dividen en chunks para aumentar el recall del RAG.

La respuesta híbrida SKU + LLM está diseñada para integrarse de forma natural.

- Licencia

Este proyecto es de propósito demostrativo para pruebas técnicas en GenAI/ML/Cloud.