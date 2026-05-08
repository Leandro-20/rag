import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

// ─── PASO 1: Cargar el documento ─────────────────────────────
const loader = new TextLoader("ejemplo.txt");
const documentos = await loader.load();

console.log("=== PASO 1: Documento cargado ===");
console.log("pageContent:", documentos[0].pageContent);
console.log("metadata:", documentos[0].metadata);

// ─── PASO 2: Cortar en chunks ─────────────────────────────────
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 30,
});

const chunks = await splitter.splitDocuments(documentos);

console.log("\n=== PASO 2: Chunks generados ===");
console.log("Cantidad de chunks:", chunks.length);
chunks.forEach((chunk, i) => {
  console.log(`\n-- Chunk ${i + 1} --`);
  console.log(chunk.pageContent);
});

// ─── PASO 3: Embeddings ───────────────────────────────────────
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434",
});

console.log("\n=== PASO 3: Probando embeddings ===");
// const vectorEjemplo = await embeddings.embedQuery("¿Qué es Node.js?")
// console.log("Dimensiones del vector:", vectorEjemplo.length)
// console.log("Primeros 5 valores:", vectorEjemplo.slice(0, 5))

// ─── PASO 4: Guardar en HNSWLib ───────────────────────────────
console.log("\n=== PASO 4: Guardando en vector store ===");

const vectorstore = await HNSWLib.fromDocuments(chunks, embeddings);
await vectorstore.save("./mi_base_vectorial");

console.log("Guardado en ./mi_base_vectorial");

// ─── VERIFICACIÓN: Búsqueda semántica ────────────────────────
console.log("\n=== VERIFICACIÓN: Búsqueda semántica ===");
const pregunta = "¿Quién creó Node.js?";
console.log("Pregunta:", pregunta);

const resultados = await vectorstore.similaritySearch(pregunta, 1);
resultados.forEach((doc, i) => {
  console.log(`\nResultado ${i + 1}:`);
  console.log(doc.pageContent);
});
