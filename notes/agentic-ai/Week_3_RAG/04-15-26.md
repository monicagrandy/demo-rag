### RAG Architecture & Vector Databases

- Vector DB operates as standalone service, separate from main application
  - Scales independently based on usage patterns
  - Shared across multiple agentic workflows/teams
  - Can run on same system if single-user, but separation recommended
- Vector DB functions like SQL/NoSQL databases but for numerical vectors
  - Available options: ChromaDB (open source), Pinecone, AWS/Google managed services
  - Facebook’s semantic search handles trillions of documents across 7.5B users
- Data storage strategy clarification:
  - No need to move data between vector DB and traditional databases
  - Vector DB can handle billions of documents efficiently
  - Data not in vector DB won’t be retrieved by RAG system

### AI Implementation Pipeline & Components

- RAG serves as integration layer connecting:
  - LLM endpoints
  - Retrieval endpoints
  - Additional services (web search, email)
- Application layer requires minimal compute when components hosted separately
- Machine learning models integrated as needed:
  - Small models (50-60KB) can stay in application layer
  - Complex models (SVM, dense trees) require separate scaling
- Microservices architecture recommended for production deployment

### Hands-On Learning & Project Recommendations

- Suggested starter projects:
  1. Tax filing chatbot with document-based Q&A
  2. LinkedIn post generator with adversarial testing across multiple LLMs
  3. RCA (Root Cause Analysis) automation bot for leadership teams
- Learning resources:
  - Kaggle has RAG project examples
  - Request recordings of RAG sessions from training program
  - Use Claude/Cursor for guidance while learning fundamentals

### Walmart’s Advanced AI Implementation

- Internal coding agent “YB” integrated across all teams:
  - Runs in JetBrains/VS Code and terminal
  - Handles repository setup, cross-repo updates, pull requests
  - Product teams build prototypes independently in single day
- “IB Desktop” provides enterprise-wide access:
  - Connected to Confluence, Outlook, Git, VPN
  - Single sign-on through “Lens” system
  - Performs complex analysis: RCA dumps, roadmap generation, bandwidth allocation
- Adoption reached 95%+ across all teams (product, analyst, business)
  - Built on Claude models with open-source harness
  - Custom skills developed for every workflow over 2 quarters