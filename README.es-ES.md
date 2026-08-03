# AHT: Ajuste automático de hiperparámetros con agentes de programación

> Indícale al agente qué optimizar. Lee tu proyecto, planifica una estrategia, ejecuta experimentos y aprende de cada resultado, mientras disfrutas de tu café.

[Versión en chino del README](README_zh.md)

**TL;DR: AHT** es una habilidad que transforma un agente de programación (como [OpenClaw](https://github.com/openclaw/openclaw), [Claude Code](https://code.claude.com/en/overview) o [OpenAI Codex](https://github.com/openai/codex)) en un investigador autónomo de ajuste de hiperparámetros para cualquier proyecto de deep learning basado en [Hydra](https://hydra.cc/).

El ajuste de hiperparámetros sigue siendo uno of los procesos más tediosos y problemáticos en la investigación en deep learning. Los métodos tradicionales de búsqueda —como la búsqueda en cuadrícula, la búsqueda aleatoria y los optimizadores bayesianos como [Optuna](https://optuna.org/)— tratan el espacio de hiperparámetros como una caja negra: muestrean configuraciones, evalúan métricas y repiten el proceso sin leer nunca una línea de código ni entender por qué una tasa de aprendizaje de 1e-3 funciona mejor que 1e-2. Por otro lado, los investigadores aportan intuición: leen el modelo, analizan las curvas de pérdida y razonan sobre qué probar a continuación. Sin embargo, esta intuición es costosa, ya que requiere horas de intervención manual y de cambiar de contexto entre experimentos.

AHT salva esta brecha. Enseña a un agente de programación a ajustar hiperparámetros como lo haría un investigador: primero lee el proyecto, luego razona sobre qué cambiar, y al mismo tiempo hereda la tenacidad de la búsqueda automatizada: ejecuta experimentos durante toda la noche, gestiona su propia cola de experimentos y se despierta cuando finaliza un trabajo de entrenamiento.

## Resumen general

AHT adopta un enfoque fundamentalmente diferente. En lugar de realizar una búsqueda ciega, equipa a un agente de programación con las herramientas para **entender** primero el proyecto y luego **razonar** sobre qué cambiar a continuación:

1. **Leer** — El agente recorre el código, analiza la jerarquía de configuración de Hydra y genera documentación estructurada (`PROJECT.md`, `HPARAM.md`) que captura la arquitectura del modelo, el pipeline de entrenamiento y los parámetros ajustables.
2. **Planificar** — Antes de que se ejecute cualquier experimento, el agente elabora una estrategia de ajuste: qué hiperparámetros priorizar, qué rangos tienen sentido dada la arquitectura y qué patrones observar.
3. **Ejecutar** — Los comandos de entrenamiento se lanzan de forma asíncrona en sesiones tmux desconectadas (tanto localmente como por SSH). El agente comprueba su finalización, estima tiempos de finalización y utiliza recordatorios cron para despertarse, sin necesidad de supervisión humana.
4. **Analizar** — Después de cada ejecución, los archivos de eventos de TensorBoard se analizan y se convierten en resúmenes estructurados de métricas escalares. El agente detecta divergencias, mesetas y sobreajuste, y registra sus hallazgos en un informe acumulativo.
5. **Aprender** — Cada decisión de ajuste posterior se basa en el historial completo de ejecuciones: overrides pasados, tendencias de métricas y el propio análisis del agente. Este bucle cerrado permite al agente refinar su estrategia con el tiempo en lugar de explorar de forma ciega.

El resultado es un proceso de ajuste iterativo y contextual que combina el rigor de la experimentación sistemática con la intuición de un investigador experimentado, y que se ejecuta de forma autónoma desde el primer experimento hasta el informe final.

### Comparación con otros enfoques de investigación/autotuning autónomos

En comparación con los enfoques de autoresearch existentes, AHT ocupa un punto muy específico en el espacio de diseño: **basado en habilidades**, **nativo de Hydra**, **poco intrusivo** y **enfocado en el ajuste**:

| Repositorio | Alcance | Como habilidad | Soporte de plataforma | Intrusión en los flujos de trabajo existentes |
| --- | --- | --- | --- | --- |
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | Optimización/general / iteración autónoma | ✅ | Claude Code | Alto |
| [ARIS ⚔️](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | Flujos de trabajo de investigación en ML | ✅ | Claude Code / Codex / OpenClaw / cualquier agente LLM | Medio |
| [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | Investigación autónoma completa (idea → artículo) | ❌ | OpenClaw / Claude Code / CLI | Alto |
| [HKUDS/ClawTeam](https://github.com/HKUDS/ClawTeam) | Orquestación multiagente para experimentos autónomos | ✅ | Claude Code / Codex / OpenClaw / nanobot / Cursor / agentes CLI personalizados | Medio |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Experimentación autónoma en ML en un pequeño repositorio de entrenamiento de LLM | ❌ | - | Es un proyecto independiente |
| [facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl) | Ajuste de hiperparámetros en RL | ❌ | Hydra | Bajo |
| **[AHT](https://github.com/zxh0916/auto-hparam-tuning)** | **Ajuste de hiperparámetros para proyectos Hydra** | ✅ | Claude Code, OpenClaw | **Bajo** |

## ✨ Características

### Comprensión del proyecto y de la configuración

AHT recorre el proyecto objetivo para identificar el script de entrada, la estructura de configuración de Hydra y los hiperparámetros ajustables, generando `PROJECT.md` y `HPARAM.md` como referencias estructuradas para posteriores decisiones de ajuste.

### Análisis de eventos de TensorBoard

AHT expone los datos escalares de TensorBoard al agente, permitiéndole detectar patrones de entrenamiento como divergencias, mesetas y sobreajuste a partir de las métricas registradas.

### Ajuste contextual con historial de ejecuciones

En cada iteración de ajuste, AHT genera un subagente con una visión general del proyecto, los overrides históricos, la estrategia de ajuste y los resultados acumulados como contexto, permitiéndole aprender de ejecuciones anteriores y tomar decisiones informadas para el próximo override.

### Ejecución asíncrona con tmux

Las ejecuciones de entrenamiento se lanzan en sesiones tmux desconectadas (tanto localmente como por SSH), lo que permite al agente comprobar su estado, estimar tiempos de finalización y establecer recordatorios cron en lugar de bloquearse.

### Historial de experimentos e informes

AHT mantiene un directorio de sesiones estructurado (`aht/yyyy-mm-dd/hh-mm-ss/`) con configuraciones, métricas y análisis por ejecución. Un script de informes integrado puede generar informes resumidos, en Markdown o HTML comparando ejecuciones.

## 🔄 Flujo de trabajo

1. **Comprender el proyecto** — Inspeccionar la estructura del proyecto y la jerarquía de configuración de Hydra; generar `PROJECT.md` y `HPARAM.md` si no existen.
2. **Comprender el comando de ejecución** — Analizar el comando de entrenamiento proporcionado por el usuario para identificar las configuraciones activas, las rutas de salida, los candidatos a métricas y los hiperparámetros relevantes.
3. **Crear una sesión** — Inicializar una sesión de ajuste con el comando base, la métrica principal y el objetivo de optimización; insertar automáticamente `- override` en la lista de valores por defecto de Hydra.
4. **Bucle de ajuste** (ejecución de referencia + hasta *N* iteraciones):
   1. Generar un subagente para decidir el mejor override basado en la estrategia y el historial de ejecuciones.
   2. Lanzar la ejecución en una sesión tmux desconectada.
   3. Comprobar el estado de la ejecución; establecer un recordatorio cron si aún está en curso.
   4. Una vez finalizada, generar un subagente para analizar el archivo de eventos de TensorBoard y actualizar el informe.
5. **Finalizar** — Presentar el informe final y la mejor configuración al usuario.

## 🚀 Inicio rápido

### Claude Code

1. Clonar el repositorio y crear enlaces simbólicos en el directorio de habilidades de Claude Code:
```bash
git clone https://github.com/zxh0916/auto-hparam-tuning.git
cd auto-hparam-tuning
pip install -r requirements.txt
# Instalación global: crear enlaces simbólicos en ~/.claude/skills
bash install_claudecode.sh
# Instalación en el proyecto: crear enlaces simbólicos en /path/to/project/.claude/skills
bash install_claudecode.sh /path/to/project
```

### OpenClaw

1. Clonar el repositorio en el directorio global de habilidades y instalar las dependencias:
```bash
cd ~/.openclaw/skills
git clone https://github.com/zxh0916/auto-hparam-tuning.git
pip install -r auto-hparam-tuning/requirements.txt
```

2. Modificar la configuración de OpenClaw:
```json
{
  "skills": {
    "load": {
      "extraDirs": [
        "~/.openclaw/skills/auto-hparam-tuning/skills"
      ]
    },
    "entries": {
      "auto-hparam-tuning": { "enabled": true },
      "aht-init": { "enabled": true }
    }
  }
}
```

### Uso

```
/auto-hparam-tuning Por favor, ajusta los hiperparámetros del proyecto "/path/to/project" en "some_remote_machine", utilizando el entorno conda remoto "some_remote_conda_env" y el entorno conda local "some_local_conda_env".
```

#### Usar diferentes modelos para los subagentes

Puedes especificar diferentes modelos para el ajuste de hiperparámetros y el análisis de resultados estableciendo variables de entorno en `openclaw.json`:
```json
{
  "env": {
    "AHT_TUNING_MODEL": "minimax/minimax-m2.5",
    "AHT_ANALYZE_MODEL": "moonshot/kimi-k2.5"
  }
}
```
Si estos valores no están establecidos, se utilizará el modelo predeterminado del agente (`agents.list[].model.primary`).

## 📝 Lista de tareas pendientes

- [ ] Añadir soporte para Codex
  - [ ] Añadir una habilidad para crear tareas cron en Codex
  - [ ] Escribir el prompt para la generación de subagentes
- [ ] Añadir soporte para Claude Code
  - [x] Escribir el prompt para la generación de subagentes
  - [x] Escribir el prompt para la creación de tareas cron
  - [ ] Establecer permisos para la habilidad
- [ ] Añadir un helper para transferir un proyecto existente no basado en Hydra a uno basado en Hydra
- [ ] Añadir soporte para especificar el modelo para los subagentes de ajuste y análisis
- [ ] ...

## 🤗 Citación

Si consideras que este proyecto es útil en tu investigación, por favor cita Hydra y AHT utilizando las siguientes entradas BibTeX:

```bibtex
@Misc{Zhang2026AHT,
  author =       {Xinhong Zhang, Weipu Zhang, Haolin Chen},
  title =        {AHT: Automatic Hyperparameter Tuning with Coding Agents using Hydra},
  howpublished = {Github},
  year =         {2026},
  url =          {https://github.com/zxh0916/auto-hparam-tuning}
}
```
```bibtex
@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```

Si tienes alguna pregunta, no dudes en crear un problema o unirte al grupo de WeChat:

<img src="imgs/wechat_group_20260324.jpg" style="zoom:25%;" />

## Historial de estrellas

<a href="https://www.star-history.com/?repos=zxh0916%2Fauto-hparam-tuning&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
   <img alt="Gráfico de historial de estrellas" src="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
 </picture>
</a>
