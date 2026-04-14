# Paper Agent

> 一个为“读论文”而设计的本地智能体：先帮你找到值得读的论文，再把你真正关心的那篇讲明白。

---

## ✨ 项目定位（Why This Project?）

很多论文助手都有两个痛点：
- 检索阶段：给出很多标题，但难以快速比较“哪篇更值得读”，且存在”幻觉“问题，许多虚假论文
- 问答阶段：RAG 直接回答，证据不足时容易出现低置信度结论

**Paper Agent** 通过多Agent协作把“找到论文 + 读懂论文”这两步做得更可靠、更顺手。

---

## 🚀 核心能力（What It Does?）

### 1) 论文检索模式（Retrieval）
- 自动识别“你寻求论文推荐”
- 调用 ArXiv 返回 Top-5 高相关论文
- 对原论文内容作初步总结
- 输出结构化候选：标题、作者、日期、摘要、PDF 链接
- 支持一键选择某篇论文进入精读

### 2) 论文精读模式（DeepRead）
- 上传 PDF 后，使用 Docling 解析文档结构
- 生成 chunk + 图表资产，写入本地 Chroma 向量库
- 判定Agent做“证据是否足够回答”判定
- 若不足，自动重写 query 进行二次检索
- 输出结构化回答，并在侧边栏展示图/表

### 3) 普通对话模式（Chat）
- 支持直接交流与解释，不强制进入论文流程
- 与检索/精读共用一个入口，由路由节点自动分流

---

## 🌟 项目亮点（Highlights）

- **路由更稳**：显式模式切换 + LLM 意图判别双保险
- **检索更准**：先抽关键词再搜，而不是整句直接检索
- **回答更实**：引入检索质量判定节点，降低“无依据回答”
- **交互更顺**：流式输出 + 可交互论文选择 + 图表侧边栏联动

---

## 🧩 技术栈（Tech Stack）

- **Orchestration**: LangGraph
- **LLM**: DeepSeek（OpenAI-compatible API）
- **UI**: Chainlit
- **PDF Parsing**: Docling + PyMuPDF
- **Vector DB**: Chroma
- **Embedding**: BAAI/bge-m3（local path）

---

## 📁 项目结构（Project Structure）

```text
Paper_Agent/
├─ app.py
├─ .chainlit/config.toml
└─ src/
   ├─ config/      # runtime settings
   ├─ graphs/      # router / retrieval / deepread / chat
   ├─ tools/       # arxiv / llm / pdf / vector / query-rewrite
   ├─ services/    # session files and storage helpers
   ├─ state/       # shared state and schema
   └─ ui/          # Chainlit handlers and renderers
```

---

## ⚙️ 快速开始（Quick Start）

### 1. 安装依赖

```bash
pip install chainlit langgraph langchain langchain-openai langchain-chroma langchain-huggingface chromadb sentence-transformers arxiv docling pymupdf
```

### 2. 配置环境变量（PowerShell 示例）

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:EMBEDDING_LOCAL_DIR="<your project>\Paper_Agent\models\bge-m3"
```

### 3. 启动

```bash
chainlit run app.py -w
```

---

## TODO

- [ ] 增加检索 rerank + 去重，提升 Top-K 质量
- [ ] 增加精读报告导出（Markdown / PDF）
- [ ] 增加基础向量数据库导入（为LLM导入行业经典论文库）

---

# 未完待续...