import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { existsSync, readdirSync, statSync } from "fs";
import { join, extname } from "path";
import { createInterface } from "readline";

const VECTOR_STORE_PATH = "./mi_base_vectorial_repo";

const EXTENSIONES = new Set([
  ".js",
  ".ts",
  ".jsx",
  ".tsx",
  ".md",
  ".txt",
  ".json",
  ".yaml",
  ".yml",
  ".css",
  ".html",
  ".sh",
]);

const EXCLUIR_CARPETAS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  "coverage",
]);

function obtenerArchivos(dir) {
  const archivos = [];
  for (const entrada of readdirSync(dir)) {
    if (EXCLUIR_CARPETAS.has(entrada)) continue;
    const ruta = join(dir, entrada);
    if (statSync(ruta).isDirectory()) {
      archivos.push(...obtenerArchivos(ruta));
    } else if (EXTENSIONES.has(extname(entrada))) {
      archivos.push(ruta);
    }
  }
  return archivos;
}

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434",
});

const llm = new Ollama({
  model: "qwen2.5-coder",
  baseUrl: "http://localhost:11434",
});

// ─── Preguntar la ruta al usuario ────────────────────────────
const rl = createInterface({ input: process.stdin, output: process.stdout });

const REPO_PATH = await new Promise((resolve) => {
  rl.question("Ruta del archivo o carpeta a indexar: ", (ruta) => {
    resolve(ruta.trim());
  });
});

if (!existsSync(REPO_PATH)) {
  console.log(`No existe la ruta: ${REPO_PATH}`);
  rl.close();
  process.exit(1);
}

// ─── ETAPA 1: Indexación ──────────────────────────────────────
// Siempre re-indexa con el archivo/carpeta que el usuario pasó
const esArchivo = statSync(REPO_PATH).isFile();
const archivos = esArchivo ? [REPO_PATH] : obtenerArchivos(REPO_PATH);

console.log(`\nArchivos encontrados (${archivos.length}):`);
archivos.forEach((a) => console.log(" -", a));

const documentos = [];
for (const archivo of archivos) {
  const loader = new TextLoader(archivo);
  const docs = await loader.load();
  documentos.push(...docs);
}

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 30,
});
const chunks = await splitter.splitDocuments(documentos);
console.log(`Total de chunks generados: ${chunks.length}`);

const vectorstore = await HNSWLib.fromDocuments(chunks, embeddings);
await vectorstore.save(VECTOR_STORE_PATH);
console.log("Base vectorial guardada en disco.");

// ─── ETAPA 2 + 3: Retrieval + Generación en loop ─────────────
const preguntar = () => {
  rl.question("\nPregunta (o 'salir'): ", async (pregunta) => {
    if (pregunta.toLowerCase() === "salir") {
      console.log("Hasta luego.");
      rl.close();
      return;
    }

    const chunks = await vectorstore.similaritySearch(pregunta, 6);
    const contexto = chunks.map((c) => c.pageContent).join("\n---\n");
    const fuentes = [...new Set(chunks.map((c) => c.metadata.source))];
    console.log("\nFuentes consultadas:", fuentes.join(", "));

    const prompt = `Usá el siguiente contexto como referencia principal para responder la pregunta.
Si la información está repartida en varios fragmentos, combinala para dar una respuesta completa.

Contexto:
${contexto}

Pregunta: ${pregunta}`;

    const respuesta = await llm.invoke(prompt);
    console.log("\nRespuesta:", respuesta);

    if (!rl.closed) preguntar();
  });
};

console.log("\nRAG listo. Podés hacer preguntas.");
preguntar();
