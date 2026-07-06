"""Generate the eval corpus, chunk it, embed it, and index it in ChromaDB.

WHAT: single shared index used by all three retrieval methods
WHY: fair comparison requires identical corpus and embeddings — only retrieval
     logic changes between baseline / rerank / MMR
"""

from __future__ import annotations

import os
from typing import Dict, List

import chromadb
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40

# ----------------------------------------------------------------------------
# Corpus text. Each document is information-dense and factual so that
# keyword-based ground truth annotation (see eval_set.py) is unambiguous.
# ----------------------------------------------------------------------------

MACHINE_LEARNING_TEXT = """Machine Learning: Foundations, Methods, and Evaluation

Machine learning is a field of artificial intelligence in which algorithms improve their performance on a task through exposure to data, rather than through explicit, hand-coded rules. Three broad paradigms dominate the field: supervised learning, unsupervised learning, and reinforcement learning, each suited to different problem structures.

Supervised learning is a machine learning paradigm in which a model learns from labeled training data, where every example is paired with the correct output. The model adjusts its internal parameters until its predictions closely match the labels provided, and once trained, it can generalize to new, unseen inputs. Common supervised tasks include classification and regression.

Unsupervised learning works with unlabeled data, seeking hidden structure without any predefined output labels. Clustering algorithms such as k-means partition data points into groups based on similarity, while dimensionality reduction techniques like principal component analysis compress high-dimensional data into fewer dimensions while preserving variance.

Reinforcement learning trains an agent to take actions in an environment in order to maximize cumulative reward. Unlike supervised learning, the agent is not told the correct action directly; instead it learns through trial and error, guided by reward signals that arrive after a sequence of decisions.

A neural network is composed of layers of interconnected neurons. Each neuron computes a weighted sum of its inputs and passes the result through a nonlinear activation function such as ReLU, sigmoid, or tanh, allowing the network to model complex, nonlinear relationships in data. Stacking many such layers creates a deep neural network.

Activation functions introduce the nonlinearity that lets neural networks approximate arbitrary functions. ReLU, or rectified linear unit, outputs zero for negative inputs and the input itself for positive inputs, and is favored in modern architectures because it avoids the vanishing-gradient problems associated with sigmoid and tanh in deep networks.

Backpropagation computes the gradient of the loss function with respect to every weight in the network by applying the chain rule of calculus, propagating error signals backward from the output layer to the input layer. Gradient descent then updates each weight in the direction that reduces the loss, with the learning rate controlling the size of each update step.

Optimizers refine plain gradient descent to converge faster and more reliably. Stochastic gradient descent updates weights using small random batches of data rather than the full dataset, while Adam combines momentum with adaptive per-parameter learning rates, making it a popular default choice for training deep networks.

Overfitting occurs when a model performs very well on training data but fails to generalize to unseen data, effectively memorizing noise rather than learning the underlying pattern. Regularization techniques such as L2 weight decay and dropout combat overfitting by penalizing overly complex models or randomly disabling neurons during training.

Dropout works by randomly setting a fraction of neuron activations to zero during each training step, forcing the network to learn redundant, robust representations rather than relying on any single neuron. At test time, dropout is disabled and activations are scaled to account for the neurons that were dropped during training.

The bias-variance tradeoff describes the tension between a model that is too simple to capture the underlying pattern, known as high bias or underfitting, and a model that is too complex and captures noise as if it were signal, known as high variance or overfitting. Good model selection balances the two.

Cross-validation, most commonly k-fold cross-validation, splits the dataset into k equally sized folds, training on k-1 folds and validating on the remaining fold, rotating through all folds to produce a robust estimate of generalization performance without needing a separate holdout set.

Precision measures the fraction of predicted positive cases that are actually correct, while recall measures the fraction of actual positive cases that the model successfully identifies. The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns, especially useful when class distributions are imbalanced.

For imbalanced classification problems, the ROC-AUC metric, area under the receiver operating characteristic curve, is often reported alongside precision and recall because accuracy alone can be misleading when one class vastly outnumbers the other.

Ensemble methods combine multiple models to improve predictive performance. Random forests build many decision trees on random subsets of data and features, then average their predictions, while gradient boosting builds trees sequentially, each one correcting the errors of the previous ensemble.

Decision trees split the feature space into regions by repeatedly asking yes-or-no questions about feature values, producing a model that is easy to interpret but prone to overfitting when grown too deep without pruning or ensemble averaging.

Support vector machines find the hyperplane that maximizes the margin between classes, and through the kernel trick can implicitly map data into higher-dimensional spaces to separate classes that are not linearly separable in the original feature space.

Feature engineering, the process of transforming raw data into informative input variables, often has as much impact on model performance as the choice of algorithm itself, since even the most powerful model cannot learn a pattern that its input features do not represent.

Hyperparameter tuning searches over settings such as learning rate, network depth, and regularization strength that are not learned directly from data but instead configured before training, commonly using grid search, random search, or Bayesian optimization.

Transfer learning reuses a model trained on one large dataset as the starting point for a related task with less data, fine-tuning some or all of its layers, which has become especially common in computer vision and natural language processing where large pretrained models are widely available.
"""

CLIMATE_SCIENCE_TEXT = """Climate Science: Mechanisms, Feedbacks, and Tipping Points

Earth's climate system is governed by a balance between incoming solar radiation and outgoing infrared radiation. Small, sustained shifts in this energy balance, caused by changes in atmospheric composition, ocean circulation, or solar output, can accumulate into large long-term changes in global temperature.

The greenhouse effect describes how gases such as carbon dioxide, methane, and water vapor trap outgoing infrared radiation in the atmosphere, warming the planet's surface above the temperature it would have without an atmosphere. Since the industrial revolution, human emissions of carbon dioxide from burning fossil fuels have significantly intensified this effect.

Radiative forcing quantifies the change in energy balance caused by a given factor, such as increased atmospheric carbon dioxide, measured in watts per square meter; positive radiative forcing warms the climate system while negative forcing, such as from reflective aerosols, cools it.

The ice-albedo feedback loop is a self-reinforcing cycle in which melting ice and snow expose darker ocean or land surfaces that absorb more sunlight than the reflective ice did, accelerating further warming and further ice loss, particularly pronounced in the Arctic.

The water vapor feedback amplifies warming because warmer air holds more water vapor, itself a potent greenhouse gas, creating a positive feedback loop where initial warming from carbon dioxide leads to more water vapor, which causes additional warming.

Permafrost thaw releases methane and carbon dioxide that has been locked in frozen soil for thousands of years, and because methane is a far more potent greenhouse gas than carbon dioxide over short timescales, this creates a dangerous tipping point where warming triggers further warming.

The Atlantic Meridional Overturning Circulation, or AMOC, is a system of ocean currents that transports warm water northward near the surface and cold water southward at depth, and its potential slowdown or collapse is considered one of the most consequential tipping points in the climate system because of its influence on regional weather patterns.

Climate tipping points are thresholds beyond which a small additional change triggers a large, often irreversible shift in the climate system, such as the collapse of an ice sheet or the dieback of a rainforest, potentially locking in consequences for centuries even if emissions are later reduced.

The West Antarctic and Greenland ice sheets hold enough frozen water to raise global sea levels by many meters if they were to collapse; satellite observations show both are already losing mass, and beyond a certain point the collapse of marine-based ice sheets may become self-sustaining and irreversible.

The carbon cycle describes the movement of carbon between the atmosphere, oceans, land, and living organisms; human fossil fuel combustion and deforestation have shifted the balance of this cycle, adding carbon to the atmosphere faster than natural sinks can absorb it.

Ocean acidification occurs as oceans absorb roughly a quarter of human carbon dioxide emissions, lowering seawater pH and making it harder for corals, shellfish, and other calcifying organisms to build their calcium carbonate skeletons and shells.

Sea level rise results from two main processes: thermal expansion of ocean water as it warms, and the addition of melted water from land-based glaciers and ice sheets, with rates of rise accelerating over recent decades according to satellite altimetry.

Extreme weather events, including heat waves, heavy precipitation, and prolonged droughts, are becoming more frequent and intense as background warming shifts the statistical distribution of possible weather outcomes toward more extreme values.

Coral reef dieback and Amazon rainforest dieback are both cited as potential tipping elements, where sustained warming or reduced rainfall could push these ecosystems past a threshold into a different, less biodiverse stable state.

Paleoclimate evidence from ice cores, drilled from Antarctica and Greenland, preserves bubbles of ancient atmosphere and allows scientists to reconstruct temperature and greenhouse gas concentrations going back hundreds of thousands of years.

Climate models, or general circulation models, simulate the physics of the atmosphere and oceans on supercomputers to project how the climate system will respond to different future emissions scenarios, and are validated by their ability to reproduce past observed climate.

Mitigation refers to efforts that reduce greenhouse gas emissions or remove carbon from the atmosphere, while adaptation refers to adjusting human systems, such as infrastructure and agriculture, to cope with climate changes that are already underway or unavoidable.

The transition to renewable energy sources such as solar and wind power, alongside electrification of transport and heating, is considered a central pillar of mitigation strategies aimed at reducing reliance on fossil fuels.

The Paris Agreement, adopted in 2015, is an international treaty in which participating countries commit to limit global warming to well below two degrees Celsius above pre-industrial levels, with an aspirational target of one and a half degrees.

A carbon budget represents the total amount of carbon dioxide that can still be emitted while keeping warming below a given threshold; reaching net-zero emissions, where remaining emissions are balanced by removals, is the point at which additional warming from carbon dioxide would largely stop.
"""

INTERNET_HISTORY_TEXT = """A History of the Internet: From ARPANET to the Mobile Web

The internet did not appear all at once; it evolved over five decades through a sequence of research projects, standards efforts, and infrastructure buildouts that together transformed a small academic experiment into a global communications system used by billions of people.

ARPANET, funded by the United States Department of Defense's Advanced Research Projects Agency, became operational in 1969 and is widely regarded as the first network to implement packet switching, breaking data into small packets that could be routed independently across the network rather than requiring a dedicated circuit.

Packet switching allows a single communication link to be shared efficiently among many simultaneous conversations, since packets from different sources can be interleaved on the same wire and reassembled at their destination, in contrast to the dedicated circuits used by the traditional telephone network.

The TCP/IP protocol suite, developed by Vint Cerf and Robert Kahn, standardized how data is broken into packets, addressed, and reassembled, allowing different, independently operated networks to interconnect into a single network of networks, which is why the resulting system came to be called the internet.

The Domain Name System, or DNS, translates human-readable domain names into the numeric IP addresses that computers use to route traffic, functioning as a distributed, hierarchical directory service that scales to billions of lookups per day without any central point of failure.

In 1989, Tim Berners-Lee, working at CERN, proposed the World Wide Web, a system of interlinked hypertext documents accessed via the internet, and went on to write the first web browser, web server, and the HTTP protocol that defines how such documents are requested and delivered.

HTML, the Hypertext Markup Language, describes the structure of a web page using nested tags, while HTTP, the Hypertext Transfer Protocol, defines the request-response cycle by which a browser asks a server for a page and the server returns it, forming the foundation of the web.

The early 1990s browser era began with Mosaic, the first widely used graphical web browser, followed by Netscape Navigator, which commercialized the browser and sparked what became known as the browser wars against Microsoft's Internet Explorer later in the decade.

Dial-up internet access, limited to the narrow bandwidth of a telephone line, gave way in the late 1990s and 2000s to broadband technologies such as digital subscriber line, or DSL, which used existing telephone wiring, and cable internet, which used the same coaxial infrastructure as cable television, both offering far higher speeds than dial-up modems.

Fiber-optic internet, which transmits data as pulses of light through glass fibers, later superseded DSL and cable in many areas, offering dramatically higher bandwidth and lower latency than copper-based broadband technologies.

The dot-com boom of the late 1990s saw a surge of investment in internet-based companies and e-commerce, followed by a sharp crash in 2000 to 2001 when many of these companies, lacking sustainable business models, failed; the survivors, such as Amazon and eBay, went on to become dominant e-commerce platforms.

Search engines such as AltaVista and later Google made the rapidly growing web navigable by indexing pages and ranking them by relevance, with Google's PageRank algorithm, which ranked pages partly by the number and quality of links pointing to them, proving especially influential.

The introduction of smartphones, especially following the 2007 launch of the iPhone, drove a rapid shift toward the mobile web, as touchscreen devices with always-on cellular data connections made browsing, email, and applications accessible away from a desktop computer, fundamentally changing how most people access the internet today.

App stores, introduced alongside modern smartphones, created a new distribution channel for software, shifting much of internet usage away from browsers and toward dedicated mobile applications for messaging, social media, and commerce.

Social media platforms such as Facebook, Twitter, and later Instagram and TikTok transformed the internet from a largely one-directional publishing medium into an interactive space built around user-generated content and algorithmically curated feeds.

Cloud computing, popularized by services such as Amazon Web Services, allows businesses to rent computing power and storage on demand rather than operating their own physical servers, underpinning much of the modern internet's infrastructure.

Net neutrality is the principle that internet service providers should treat all data on their networks equally, without charging differently or deliberately slowing traffic based on the content, platform, or application generating it, and has been the subject of ongoing regulatory debate.

The transition from IPv4 to IPv6 was driven by the exhaustion of the roughly four billion addresses available under IPv4, with IPv6 providing a vastly larger address space to accommodate the growing number of internet-connected devices.

Wireless standards evolved through successive cellular generations, from 2G digital voice and basic data, through 3G and 4G LTE mobile broadband, to 5G, alongside Wi-Fi standards that brought high-speed wireless networking to homes and offices.

Modern internet infrastructure relies heavily on content delivery networks, or CDNs, and geographically distributed data centers to cache and serve content close to users, reducing latency and allowing services to handle massive, globally distributed traffic loads.
"""

LLM_SYSTEMS_TEXT = """Large Language Model Systems: Architecture, Training, and Applications

Modern large language models are built on a sequence of innovations spanning model architecture, training techniques, and system design that together allow a single model to generate fluent text, follow instructions, retrieve external knowledge, and take actions through tools.

The transformer architecture, introduced in 2017, relies on a self-attention mechanism that allows the model to weigh the relevance of every other token in the input when computing the representation of a given token, capturing long-range dependencies far more effectively than earlier recurrent architectures.

Multi-head attention runs several attention computations in parallel, each potentially focusing on different relationships between tokens, such as syntactic structure or semantic similarity, and combines their outputs to form a richer representation of the input sequence.

Pretraining exposes a transformer to enormous quantities of text, typically trained to predict the next token given the preceding context, and it is through this simple objective, applied at massive scale, that the model acquires broad knowledge of language, facts, and reasoning patterns.

Fine-tuning takes a pretrained model and further trains it on a smaller, more specific dataset, adjusting its weights to specialize in a particular task, domain, or behavior without needing to train a model from scratch.

Reinforcement learning from human feedback, or RLHF, aligns a pretrained language model with human preferences by training a reward model on human rankings of candidate responses, then using reinforcement learning to fine-tune the language model to produce outputs the reward model scores highly.

Instruction tuning fine-tunes a pretrained model on a dataset of instructions paired with desired responses, teaching the model to follow natural-language commands rather than simply continuing text in the style of its pretraining data.

Retrieval-augmented generation, or RAG, addresses the problem of hallucination, where a language model generates fluent but factually incorrect text, by retrieving relevant documents from an external knowledge source and inserting them into the model's context before it generates a response, grounding the output in retrieved evidence.

Vector search underpins most RAG systems: documents are converted into numerical embedding vectors that capture semantic meaning, stored in a vector database, and retrieved by finding the vectors most similar to the embedding of an incoming query.

Hallucination occurs because a language model is trained to produce plausible, fluent continuations of text rather than to verify factual accuracy, so it can state incorrect information confidently and with the same fluent tone as correct information, making hallucinations difficult for users to detect without independent verification.

LLM-based agents extend a language model beyond text generation by giving it the ability to call external tools, such as search engines, calculators, or code interpreters, and to use the results of those tool calls to decide on subsequent actions, enabling multi-step tasks that a single forward pass could not accomplish.

Tool calling, sometimes called function calling, works by having the model output a structured request specifying which tool to invoke and with what arguments, which the surrounding system then executes and feeds the result back into the model's context.

Chain-of-thought reasoning prompts a model to produce intermediate reasoning steps before arriving at a final answer, which has been shown to improve performance on tasks that require multi-step arithmetic, logic, or planning compared to asking for a direct answer.

The context window is the maximum number of tokens a model can attend to at once, encompassing both the prompt and the generated output, and its size determines how much retrieved evidence, conversation history, or document content can be provided to the model at inference time.

Embeddings are dense numerical vectors produced by a model to represent the meaning of a piece of text, positioned in a high-dimensional space such that semantically similar pieces of text have vectors that are close together under a distance metric such as cosine similarity.

Choosing between fine-tuning and prompting a model depends on the task: prompting, including few-shot examples in the prompt, is faster to iterate on and requires no training, while fine-tuning can more deeply and durably change a model's behavior at the cost of additional data and compute.

Prompt engineering is the practice of carefully designing the instructions, examples, and context given to a language model in order to elicit more accurate, relevant, or well-formatted outputs without changing the model's underlying weights.

Evaluating large language models relies on a mix of automated benchmarks covering knowledge, reasoning, and coding tasks, alongside human evaluation and LLM-as-judge approaches, since many desirable qualities such as helpfulness and coherence are difficult to capture with a single automatic metric.

Safety and alignment research studies how to make language models more honest, harmless, and aligned with human intent, addressing risks such as generating harmful content, being manipulated through adversarial prompts, or pursuing goals misaligned with what users actually want.

Multimodal models extend the transformer architecture to process inputs beyond text, such as images or audio, by converting them into token-like representations that can be processed alongside text tokens within the same attention mechanism.
"""

DATABASE_SYSTEMS_TEXT = """Database Systems: Relational Foundations, Transactions, and Modern Alternatives

Database management systems organize, store, and retrieve data reliably and efficiently, and have evolved from early relational systems into a diverse ecosystem that includes document stores, key-value stores, graph databases, and, more recently, vector databases.

The relational model, introduced by Edgar Codd, organizes data into tables of rows and columns, with each table representing an entity type and each row representing a single record; Structured Query Language, or SQL, is the standard language used to define, query, and modify data stored in relational tables.

ACID is an acronym describing four properties that guarantee reliable database transactions: atomicity ensures a transaction either completes entirely or has no effect at all, consistency ensures a transaction moves the database from one valid state to another, isolation ensures concurrent transactions do not interfere with each other, and durability ensures committed changes survive a crash.

Atomicity, one of the four ACID properties, guarantees that a multi-step operation such as transferring money between two bank accounts either fully succeeds, updating both the debited and credited balances, or fully fails and leaves both balances unchanged, with no possibility of a partial update.

Most relational databases speed up lookups using an index, commonly implemented as a B-tree, a balanced tree structure that keeps data sorted and allows searches, insertions, and deletions in logarithmic time, avoiding the need to scan every row in a table.

The query optimizer is a component of a relational database that examines a submitted SQL query and chooses an efficient execution plan, deciding which indexes to use and in what order to join tables, based on statistics about the size and distribution of the underlying data.

Normalization organizes relational tables to reduce data redundancy and avoid update anomalies, typically by decomposing a table into smaller related tables connected by foreign keys, following a series of normal forms that progressively eliminate different kinds of redundancy.

NoSQL databases relax the rigid schema and relational structure of traditional databases to better handle unstructured or rapidly changing data, and are broadly categorized into document databases that store JSON-like documents, key-value stores that map simple keys to arbitrary values, and graph databases that model data as nodes and relationships.

Document databases, such as MongoDB, store records as JSON-like documents that can have nested fields and varying structure from one document to the next, making them well suited to applications where the shape of the data changes frequently or is not known in advance.

Key-value stores, such as Redis, provide extremely fast reads and writes by mapping a simple key directly to a value, often held in memory, and are commonly used for caching, session storage, and other latency-sensitive workloads rather than as a system of record.

Graph databases model data explicitly as nodes and the relationships, or edges, between them, making them well suited to queries that traverse many-to-many relationships, such as social networks, recommendation engines, or fraud detection.

The CAP theorem states that a distributed database can provide at most two of three guarantees simultaneously: consistency, availability, and partition tolerance, forcing distributed system designers to make explicit tradeoffs, especially when a network partition occurs.

Sharding splits a large database horizontally across multiple servers, with each shard holding a subset of the rows, while replication copies data across multiple servers to improve availability and read throughput; distributed databases commonly combine both techniques.

Vector databases store high-dimensional embedding vectors produced by machine learning models and support similarity search, retrieving the vectors nearest to a query vector under a distance metric such as cosine similarity or Euclidean distance.

Because exact nearest-neighbor search becomes prohibitively slow at large scale, vector databases typically use approximate nearest neighbor algorithms such as HNSW, hierarchical navigable small world graphs, which build a multi-layer graph structure that allows searches to quickly navigate to a vector's approximate nearest neighbors while sacrificing a small amount of accuracy for large gains in speed.

Transactions and concurrency control mechanisms, such as locking and multi-version concurrency control, allow multiple users to read and write a database at the same time while preserving the isolation guarantee that each transaction appears to execute as if it were the only one running.

Data warehouses are databases optimized for analytical queries over large historical datasets, typically populated through extract-transform-load pipelines from many operational source systems, and are queried far less frequently but with much larger, more complex queries than operational databases.

Column-oriented storage, used by many analytical databases, stores each column of a table contiguously on disk rather than each row, which dramatically speeds up queries that aggregate or scan a small number of columns across many rows.

Caching layers, often built on key-value stores, sit in front of a primary database to serve frequently requested data from fast memory rather than repeatedly querying slower disk-backed storage, reducing both latency and load on the primary database.

Database backup and recovery procedures, including periodic full backups and continuous write-ahead logging, allow a database to be restored to a consistent state after hardware failure, data corruption, or accidental deletion.
"""

DOCUMENTS: Dict[str, str] = {
    "machine_learning.txt": MACHINE_LEARNING_TEXT,
    "climate_science.txt": CLIMATE_SCIENCE_TEXT,
    "internet_history.txt": INTERNET_HISTORY_TEXT,
    "llm_systems.txt": LLM_SYSTEMS_TEXT,
    "database_systems.txt": DATABASE_SYSTEMS_TEXT,
}


def generate_corpus(output_dir: str) -> List[str]:
    """Write the 5 corpus documents to output_dir and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, text in DOCUMENTS.items():
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------
# Recursive character splitter (chunk_size=400, overlap=40), no LangChain.
# ----------------------------------------------------------------------------

def _merge_splits(splits: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for s in splits:
        if len(current) + len(s) <= chunk_size:
            current += s
            continue
        if current:
            chunks.append(current)
        overlap_text = current[-chunk_overlap:] if chunk_overlap and current else ""
        current = overlap_text + s
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap:]
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int,
                      separators: List[str]) -> List[str]:
    sep = separators[-1]
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    splits = text.split(sep) if sep else list(text)

    final_chunks: List[str] = []
    good_splits: List[str] = []
    for i, s in enumerate(splits):
        piece = s + sep if sep and i < len(splits) - 1 else s
        if len(piece) < chunk_size:
            good_splits.append(piece)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, chunk_size, chunk_overlap))
                good_splits = []
            remaining = separators[separators.index(sep) + 1:]
            if remaining:
                final_chunks.extend(_recursive_split(piece, chunk_size, chunk_overlap, remaining))
            else:
                final_chunks.append(piece)
    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, chunk_size, chunk_overlap))
    return [c for c in final_chunks if c.strip()]


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks, tracking each chunk's char_start."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

    chunks = []
    search_pos = 0
    for chunk_text in raw_chunks:
        probe = chunk_text[:30].strip()
        idx = text.find(probe, max(0, search_pos - chunk_overlap)) if probe else search_pos
        if idx == -1:
            idx = search_pos
        chunks.append({"text": chunk_text, "char_start": idx})
        search_pos = idx + len(chunk_text)
    return chunks


def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index(corpus_paths: List[str], client: OpenAI) -> chromadb.Collection:
    """Chunk, embed, and index the corpus into a persistent ChromaDB collection.

    Idempotent: if the collection already has documents, skip re-indexing.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("eval_corpus")

    if collection.count() > 0:
        print("Index exists, skipping")
        return collection

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[Dict] = []

    for path in corpus_paths:
        doc_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_document(text)
        for chunk_id, chunk in enumerate(chunks):
            all_ids.append(f"{doc_name}_{chunk_id}")
            all_docs.append(chunk["text"])
            all_metadatas.append({
                "doc_name": doc_name,
                "chunk_id": chunk_id,
                "char_start": chunk["char_start"],
            })

    batch_size = 20
    for start in range(0, len(all_docs), batch_size):
        end = start + batch_size
        batch_texts = all_docs[start:end]
        embeddings = _embed_batch(client, batch_texts)
        collection.add(
            ids=all_ids[start:end],
            documents=batch_texts,
            metadatas=all_metadatas[start:end],
            embeddings=embeddings,
        )

    print(f"Indexed {len(all_docs)} chunks from {len(corpus_paths)} documents")
    return collection
