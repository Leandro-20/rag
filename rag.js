import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { OllamaEmbeddings } from "@langchain/ollama"
import { Ollama } from "@langchain/ollama"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"
import { existsSync } from "fs"
import { createInterface } from "readline"

const VECTOR_STORE_PATH = "./mi_base_vectorial"

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434",
})

const llm = new Ollama({
  model: "qwen2.5-coder",
  baseUrl: "http://localhost:11434",
})

// ─── ETAPA 1: Indexación ──────────────────────────────────────
// Solo indexa si la base vectorial no existe todavía
let vectorstore

if (existsSync(VECTOR_STORE_PATH)) {
  console.log("Base vectorial encontrada, cargando...")
  vectorstore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings)
} else {
  console.log("Base vectorial no encontrada, indexando...")

  const loader = new TextLoader("ejemplo.txt")
  const documentos = await loader.load()

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 30,
  })
  const chunks = await splitter.splitDocuments(documentos)

  vectorstore = await HNSWLib.fromDocuments(chunks, embeddings)
  await vectorstore.save(VECTOR_STORE_PATH)

  console.log(`Indexados ${chunks.length} chunks y guardados en disco.`)
}

// ─── ETAPA 2 + 3: Retrieval + Generación en loop ─────────────
const rl = createInterface({ input: process.stdin, output: process.stdout })

const preguntar = () => {
  rl.question("\nPregunta (o 'salir'): ", async (pregunta) => {
    if (pregunta.toLowerCase() === "salir") {
      console.log("Hasta luego.")
      rl.close()
      return
    }

    const chunks = await vectorstore.similaritySearch(pregunta, 2)
    const contexto = chunks.map(c => c.pageContent).join("\n---\n")

    const prompt = `Usá solo el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, decí "No lo sé".

Contexto:
${contexto}

Pregunta: ${pregunta}`

    const respuesta = await llm.invoke(prompt)
    console.log("\nRespuesta:", respuesta)

    if (!rl.closed) preguntar()
  })
}

console.log("\nRAG listo. Podés hacer preguntas sobre el documento.")
preguntar()
