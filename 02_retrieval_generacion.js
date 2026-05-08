import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

// ─── ETAPA 2: Retrieval ───────────────────────────────────────
// Cargamos la base vectorial que ya indexamos en el paso 1
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434",
});

const vectorstore = await HNSWLib.load("./mi_base_vectorial", embeddings);

// La pregunta del usuario
const pregunta = "¿Quién creó Node.js y en qué año?";

// Buscamos los 2 chunks más relevantes para esa pregunta
const chunks = await vectorstore.similaritySearch(pregunta, 1);

console.log("=== ETAPA 2: Chunks recuperados ===");
chunks.forEach((chunk, i) => {
  console.log(`\nChunk ${i + 1}:`);
  console.log(chunk.pageContent);
});

// ─── ETAPA 3: Generación ──────────────────────────────────────
// Armamos el contexto juntando los chunks recuperados
const contexto = chunks.map((c) => c.pageContent).join("\n---\n");

// Construimos el prompt con el contexto y la pregunta
const prompt = `Usá solo el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, decí "No lo sé".

Contexto:
${contexto}

Pregunta: ${pregunta}`;

console.log("\n=== ETAPA 3: Prompt enviado al LLM ===");
console.log(prompt);

// Llamamos a Ollama con el modelo local
const llm = new Ollama({
  model: "qwen2.5-coder",
  baseUrl: "http://localhost:11434",
});

const respuesta = await llm.invoke(prompt);

console.log("\n=== RESPUESTA DEL LLM ===");
console.log(respuesta);
