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

# MistralLoRA — MedQuAD (LoRA r=32)

El modelo `MistralLoRA — MedQuAD (LoRA r=32)` corresponde a una versión afinada de `mistralai/Mistral-7B-Instruct-v0.3` utilizando el conjunto de datos médico `lavita/MedQuAD`.  
El proceso de afinamiento aplicó la técnica `LoRA (Low-Rank Adaptation)` mediante `PEFT (Parameter-Efficient Fine-Tuning)` con cuantización `NF4 en 4 bits` implementada a través de `bitsandbytes`.  
El objetivo principal consiste en generar respuestas médicas breves, coherentes y fundamentadas en fuentes verificables, manteniendo una relación eficiente entre costo computacional y calidad de generación.

Este trabajo forma parte de la investigación documentada en:  
*Medrano Cerdas, J. L. (2025). _MistralLoRA — MedQuAD (LoRA r=32): Afinamiento eficiente de parámetros para generación de texto médico._ Hugging Face Hub.*

Disponible en: [https://huggingface.co/medranojl/MistralLoRAr32](https://huggingface.co/medranojl/MistralLoRAr32)


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


## Resultados del entrenamiento

El modelo alcanzó una pérdida de validación (`eval_loss`) de *0.7617* y una perplejidad final de *2.13*, logrando un desempeño robusto y estable durante el proceso de ajuste fino.  
La siguiente figura muestra la evolución de la pérdida durante las etapas de entrenamiento y validación del modelo `LoRA r=32`:

![Curva de entrenamiento LoRA r=32](https://huggingface.co/medranojl/MistralLoRAr32/resolve/main/assets/training_curve_r32.png)

*Figura 1.* Evolución de la pérdida de entrenamiento y validación del modelo MistralLoRA r=32.


## Evaluación en generación de texto

El modelo fue evaluado aplicando distintas estrategias de decodificación sobre el conjunto de prueba (`test`) del dataset `MedQuAD`.  
Las métricas empleadas incluyen `ROUGE-L` (coherencia semántica), `BERTScore` (factualidad), `Distinct-n` (diversidad léxica) y `Repetition Ratio` (repetición de n-gramas).  

| Estrategia | tokens_len | distinct_1 | distinct_2 | repetition_r3 | rougeL | bertscore_f1 |
|:------------|:-----------:|:-----------:|:-----------:|:---------------:|:--------:|:---------------:|
| `beam_4` | 255.83 | 0.5779 | 0.9026 | 0.0000 | 0.2232 | 0.8580 |
| `topp_0.9` | 255.67 | 0.6140 | 0.9123 | 0.0033 | 0.2432 | 0.8571 |
| `temp_1.2` | 256.00 | 0.6185 | 0.9203 | 0.0020 | 0.2223 | 0.8520 |
| `greedy` | 255.67 | 0.5587 | 0.8730 | 0.0079 | 0.2078 | 0.8514 |
| `temp_0.9` | 256.33 | 0.6417 | 0.9243 | 0.0007 | 0.2240 | 0.8497 |
| `topk_50` | 256.00 | 0.6490 | 0.9229 | 0.0007 | 0.2166 | 0.8491 |

### Interpretación de los resultados

El análisis comparativo evidencia cómo la estrategia de decodificación influye directamente en las métricas de coherencia, diversidad y factualidad.  
Cada enfoque equilibra de forma diferente la precisión semántica y la variedad léxica, lo cual permite adaptar el modelo según los objetivos específicos de generación textual.

- La estrategia `beam_4` obtuvo el valor más alto en `BERTScore`, indicando mayor factualidad y consistencia semántica.  
- La configuración `topp_0.9` alcanzó la puntuación más alta en `ROUGE-L`, mostrando una coherencia superior con las respuestas de referencia.  
- El método `temp_0.9` destacó por su mayor diversidad léxica (`distinct-2 = 0.9243`), generando textos más variados y naturales.  
- Nuevamente, `beam_4` mostró la menor tasa de repetición (`repetition_r3 = 0.0000`), evidenciando estabilidad gramatical y control sintáctico.  

En conjunto, los resultados confirman que `MistralLoRA r=32` ofrece un equilibrio sólido entre precisión, coherencia y diversidad, destacando especialmente en escenarios que requieren respuestas informativas con bajo nivel de redundancia.

## Usos

### Uso directo
El modelo está diseñado para tareas de *preguntas y respuestas médicas (Q&A)* basadas en texto, generando respuestas fundamentadas en información validada.  
Ejemplos de aplicación:
- Asistentes médicos virtuales  
- Sistemas de recuperación de información en salud  
- Plataformas educativas en medicina  

### Uso en proyectos derivados
El modelo puede ser empleado como base para experimentos de generación médica multilingüe o tareas derivadas como resumen de textos, extracción de términos clínicos o clasificación semántica.

### Usos no recomendados
El modelo fue desarrollado con fines académicos y **no debe utilizarse** para:
- Emitir diagnósticos o recomendaciones clínicas  
- Sustituir la evaluación de un profesional médico  
- Procesar información sensible o confidencial  

## Sesgos, riesgos y limitaciones
Aunque el dataset `MedQuAD` contiene información verificada, las respuestas generadas pueden incluir errores o interpretaciones parciales.  
El modelo depende fuertemente del contexto de la pregunta y no reemplaza la revisión médica humana.

### Recomendaciones
- Supervisar todas las salidas generadas antes de su uso en entornos reales.  
- Emplear el modelo únicamente con fines de investigación o educativos.  
- Validar las respuestas con fuentes médicas oficiales.  


## Ejemplo de uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("medranojl/MistralLoRAr32", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("medranojl/MistralLoRAr32")

prompt = "What are the main complications of diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training/Validation Metrics (LoRA r=32) — 2025-11-02 07:57:11

| Step | Epoch | Training Loss | Validation Loss |
|-----:|-----:|--------------:|----------------:|
| 100 | 0.07 | 0.916400 |  |
| 200 | 0.14 | 0.797800 |  |
| 300 | 0.21 | 0.821200 |  |
| 400 | 0.28 | 0.806400 |  |
| 500 | 0.35 | 0.799300 |  |
| 600 | 0.42 | 0.780000 |  |
| 700 | 0.49 | 0.775600 |  |
| 800 | 0.56 | 0.743300 |  |
| 900 | 0.63 | 0.757700 |  |
| 1000 | 0.70 | 0.757400 |  |
| 1100 | 0.77 | 0.756400 |  |
| 1200 | 0.84 | 0.743100 |  |
| 1300 | 0.91 | 0.747400 |  |
| 1400 | 0.98 | 0.745200 |  |
| 1436 | 1.00 |  | 0.721013 |
| 1500 | 1.04 | 0.686400 |  |
| 1600 | 1.11 | 0.672900 |  |
| 1700 | 1.18 | 0.655800 |  |
| 1800 | 1.25 | 0.680600 |  |
| 1900 | 1.32 | 0.645500 |  |
| 2000 | 1.39 | 0.684100 |  |
| 2100 | 1.46 | 0.653600 |  |
| 2200 | 1.53 | 0.645700 |  |
| 2300 | 1.60 | 0.634400 |  |
| 2400 | 1.67 | 0.671100 |  |
| 2500 | 1.74 | 0.638600 |  |
| 2600 | 1.81 | 0.633000 |  |
| 2700 | 1.88 | 0.637000 |  |
| 2800 | 1.95 | 0.623200 |  |
| 2872 | 2.00 |  | 0.700253 |
| 2900 | 2.02 | 0.620200 |  |
| 3000 | 2.09 | 0.533300 |  |
| 3100 | 2.16 | 0.559500 |  |
| 3200 | 2.23 | 0.553000 |  |
| 3300 | 2.30 | 0.571400 |  |
| 3400 | 2.37 | 0.543400 |  |
| 3500 | 2.44 | 0.542100 |  |
| 3600 | 2.51 | 0.547500 |  |
| 3700 | 2.58 | 0.530400 |  |
| 3800 | 2.65 | 0.514300 |  |
| 3900 | 2.72 | 0.549400 |  |
| 4000 | 2.79 | 0.550600 |  |
| 4100 | 2.86 | 0.534900 |  |
| 4200 | 2.93 | 0.545300 |  |
| 4300 | 2.99 | 0.520600 |  |
| 4308 | 3.00 |  | 0.710358 |

---
# MistralLoRA — MedQuAD (LoRA r=32)
## Resultados del entrenamiento
| Aspecto | Descripción |
|----------|-------------|
| Modelo base | `mistralai/Mistral-7B-Instruct-v0.3` |
| Dataset | `lavita/MedQuAD` (división 70/15/15) |
| Quantization | 4-bit NF4 (*bitsandbytes*) |
| Método | LoRA (r=32, α=64, dropout=0.05) |
| Objetivo | Generación de respuestas médicas breves y verificables |
| Mejor métrica (`eval_loss`) | 0.700253 |
| Perplejidad final | 2.04 |
