# RAG con Node.js + Ollama

RAG (Retrieval-Augmented Generation) local y gratuito usando LangChain.js, HNSWLib y Ollama.

## ¿Cómo funciona?

```
Documento → Chunks → Embeddings → Base vectorial
                                        ↓
Pregunta → Vector → Búsqueda → Contexto → LLM → Respuesta
```

1. **Indexación** — el documento se divide en chunks, cada uno se convierte en un vector y se guarda en disco
2. **Retrieval** — la pregunta se convierte en vector y se buscan los chunks más similares
3. **Generación** — los chunks encontrados se mandan al LLM junto con la pregunta para generar una respuesta

## Requisitos

- Node.js 18+
- [Ollama](https://ollama.com) instalado y corriendo

```bash
ollama pull nomic-embed-text   # modelo de embeddings
ollama pull qwen2.5-coder      # modelo de generación
```

## Instalación

```bash
npm install
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `01_indexacion.js` | Ejemplo paso a paso de la etapa de indexación |
| `02_retrieval_generacion.js` | Ejemplo de retrieval y generación por separado |
| `rag.js` | Pipeline completo con un archivo de texto fijo |
| `rag_repo.js` | Pipeline completo con soporte para cualquier archivo o carpeta |

## Uso

### Pipeline simple (usa `ejemplo.txt`)

```bash
node rag.js
```

### Pipeline con tu propio archivo o carpeta

```bash
node rag_repo.js
```

Te va a pedir la ruta:

```
Ruta del archivo o carpeta a indexar: /ruta/a/tu/proyecto
```

Después podés hacer preguntas sobre el contenido. Escribí `salir` para terminar.

### Re-indexación inteligente

`rag_repo.js` calcula un hash MD5 de los archivos. Si el contenido no cambió desde la última vez, carga la base vectorial existente sin re-indexar.

## Stack

- **LangChain.js** — loaders, splitter y pipeline
- **HNSWLib** — base de datos vectorial local (sin servidor)
- **Ollama** — embeddings (`nomic-embed-text`) y LLM (`qwen2.5-coder`)
