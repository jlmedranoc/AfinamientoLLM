---
language: es
library_name: transformers
tags:
- lora
- peft
- mistral
- medical
- medquad
- causal-lm
- text-generation
license: apache-2.0
pipeline_tag: text-generation
base_model: mistralai/Mistral-7B-Instruct-v0.3
datasets:
- lavita/MedQuAD
---

# MistralLoRA — MedQuAD (LoRA r=4)

El modelo `MistralLoRA — MedQuAD (LoRA r=4)` corresponde a una versión afinada de `mistralai/Mistral-7B-Instruct-v0.3` utilizando el conjunto de datos médico `lavita/MedQuAD` y el proceso de afinamiento aplicó la técnica `LoRA (Low-Rank Adaptation)` mediante `PEFT (Parameter-Efficient Fine-Tuning)` con cuantización `NF4 en 4 bits` implementada a través de `bitsandbytes`, el propósito principal consiste en generar respuestas médicas breves, coherentes y fundamentadas en fuentes verificables.

Este trabajo forma parte de la investigación documentada en:  
*Medrano Cerdas, J. L. (2025). _MistralLoRA — MedQuAD (LoRA r=4): Afinamiento eficiente de parámetros para generación de texto médico._ Hugging Face Hub.*

Disponible en: [https://huggingface.co/medranojl/MistralLoRAr4](https://huggingface.co/medranojl/MistralLoRAr4)

---

## Detalles del modelo

### Descripción general
- Desarrollado por: José Luis Medrano Cerdas ([medranojl](https://huggingface.co/medranojl))  
- Tipo de modelo: `Causal Language Model (AutoModelForCausalLM)`  
- Idioma: Inglés  
- Finetuning de: `mistralai/Mistral-7B-Instruct-v0.3`  
- Licencia: Apache 2.0  
- Frameworks: `Transformers`, `PEFT`, `bitsandbytes`  
- Hardware utilizado: `NVIDIA A100 (40 GB VRAM)`  
- Propósito: generación de texto médico verificable (Q&A)  

---

## Resultados del entrenamiento

El modelo alcanzó una pérdida de validación (`eval_loss`) de *0.7526* y una perplejidad final de *2.14*, manteniendo un equilibrio entre factualidad y diversidad léxica. La siguiente figura muestra la evolución de la pérdida durante el proceso de entrenamiento y validación del modelo `LoRA r=4`:

![Curva de entrenamiento LoRA r=4](https://huggingface.co/medranojl/MistralLoRAr4/resolve/main/assets/training_curve_r4.png)

*Figura 1.* Evolución del entrenamiento y validación del modelo MistralLoRA r=4.

---

## Evaluación en generación de texto

El modelo fue evaluado aplicando diferentes estrategias de decodificación sobre el conjunto de prueba (`test`) del dataset `MedQuAD`. Las métricas consideradas incluyen `ROUGE-L` (coherencia semántica), `BERTScore` (factualidad), `Distinct-n` (diversidad léxica) y `Repetition Ratio` (repetición de n-gramas) y los resultados se muestran en la siguiente tabla.

| Estrategia | tokens_len | distinct_1 | distinct_2 | repetition_r3 | rougeL | bertscore_f1 |
|:------------|:-----------:|:-----------:|:-----------:|:---------------:|:--------:|:---------------:|
| `greedy` | 256.00 | 0.5736 | 0.8837 | 0.0007 | 0.2227 | 0.8560 |
| `topp_0.9` | 255.67 | 0.6023 | 0.8985 | 0.0000 | 0.2321 | 0.8537 |
| `temp_1.2` | 256.17 | 0.6297 | 0.9111 | 0.0052 | 0.2214 | 0.8534 |
| `topk_50` | 256.00 | 0.6380 | 0.9275 | 0.0000 | 0.2303 | 0.8520 |
| `temp_0.9` | 256.00 | 0.6107 | 0.9170 | 0.0000 | 0.1878 | 0.8468 |
| `beam_4` | 255.83 | 0.5772 | 0.9032 | 0.0000 | 0.1747 | 0.8386 |

Interpretación de los resultados:

El análisis de las estrategias de decodificación evidencia diferencias relevantes en el comportamiento del modelo según el método empleado para la generación del texto. Las métricas de factualidad, coherencia y diversidad muestran que cada enfoque prioriza distintos aspectos del lenguaje, lo que permite seleccionar la estrategia más adecuada según el propósito requerido:

- La estrategia `greedy` obtuvo el valor más alto de `BERTScore`, lo que indica una mayor precisión semántica y consistencia factual con las respuestas de referencia.  
- La configuración `topp_0.9` alcanzó la mejor puntuación en `ROUGE-L`, reflejando una mayor coherencia estructural y alineación gramatical con el texto esperado.  
- El método `topk_50` destacó en diversidad léxica (`distinct-2`), generando respuestas más variadas sin comprometer significativamente la coherencia.  
- Finalmente, `topp_0.9` presentó la menor tasa de repetición, lo que sugiere un balance adecuado entre creatividad y control sintáctico.  

En conjunto, los resultados confirman que el modelo mantiene un equilibrio sólido entre precisión semántica, coherencia textual y diversidad léxica, adaptándose eficazmente a distintas estrategias de generación.

---

## Usos

### Uso directo
El modelo está preparado para tareas del tipo *preguntas y respuestas médicas*, generando texto basado en información validada.  
Ejemplos de aplicación:
- Asistentes médicos virtuales  
- Sistemas de búsqueda semántica en salud  
- Herramientas de educación médica automatizada  

### Uso en proyectos derivados
Puede servir como base para modelos multilingües o especializados, incluyendo tareas de resumen, extracción de conceptos o clasificación de textos médicos.

### Usos no recomendados
El modelo fue desarrollado en el marco de una investigación académica y *no debe utilizarse* para:
- Emitir diagnósticos o recomendaciones médicas directas  
- Sustituir la opinión de un profesional de la salud  
- Procesar datos personales sensibles  

---

## Sesgos, riesgos y limitaciones

Aunque el conjunto de datos `MedQuAD` incluye información médica validada, el modelo puede generar respuestas parciales o interpretaciones incorrectas.  
La precisión depende de la calidad y contexto de las preguntas, por lo que los resultados no deben considerarse consejos médicos.

### Recomendaciones
- Incluir revisión humana de las respuestas generadas.  
- Evitar el uso en entornos clínicos automatizados.  
- Validar los resultados con fuentes médicas oficiales.  

---

## Ejemplo de uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("medranojl/MistralLoRAr4", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("medranojl/MistralLoRAr4")

prompt = "What are the common symptoms of anemia?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
