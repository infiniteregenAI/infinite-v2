# InfiniteRegen

The foundational core for backrooms and swarms, bringing together the smartest KOLs to revolutionize capital allocation, DAOs, funds, and collaborative ecosystems. Stay tuned as we release some good updates coming days. Thanks

## Project Status

### Completed ✅
- Basic agent profile creation and management
- Document upload and processing system
- Agent knowledge base integration
- Inter-agent conversation system
- Bucket creation and management
- Document generation and summarization
- Core business logic implementation
- Basic API structure
- Clerk authentication integration
- Team management system
- Reserved agents knowledge base

### In Progress 🚧
- FastAPI backend completion
- Agent performance metrics
- Enhanced error handling and logging
- Advanced workflow management
- Tools integration system

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment support
- Clerk account for authentication

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/infiniteregenAI/infinite-v2.git
   cd infinite-v2
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to create a new `.env` file:
   ```bash
   cp .env.example .env
   ```
   - Update the following variables in your `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   DB_URL=postgres_db_url
   CLERK_SECRET_KEY=your_clerk_secret_key
   CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
   ```

## Project Structure

```
InfiniteRegen/
├── middleware/                # Authentication and security
│   ├── clerk_middleware.py    # Clerk authentication integration
│   └── __init__.py
│
├── routes/                    # API endpoints
│   ├── agents_router.py       # Agent management endpoints
│   ├── teams_router.py        # Team management endpoints
│   └── tools_router.py        # Tools integration endpoints
│
├── schemas/                   # Data validation schemas
│   ├── agents_schema.py       # Agent data schemas
│   └── teams_schema.py        # Team data schemas
│
├── tmp/                       # Temporary database files
│   ├── agents_sessions.db     # Agent session storage
│   ├── agents.db             # Agent data storage
│   ├── teams_sessions.db     # Team session storage
│   └── workflows.db          # Workflow data storage
│
├── utils/                     # Utility functions
│   ├── agent_manager.py       # Agent management utilities
│   ├── constants.py          # System constants
│   ├── reserved_agents.py    # Pre-configured agent definitions
│   └── reserved_agents_knowledge_base/  # Knowledge base for reserved agents
│
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── agents.json              # Agent configuration template
├── teams.json              # Team configuration template
├── main.py                 # Main application entry
├── phi_server.py           # PHI-compliant server implementation
└── requirements.txt        # Project dependencies
```

## Running the Application

### Run Main Application
```bash
uvicorn main:app --reload
```

Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Run PHI-Compliant Server
```bash
uvicorn phi_server:app --reload
```

## Features

### 1. **Agent Management**
- Create and configure custom agents
- Access pre-configured reserved agents
- Manage agent knowledge bases
- Monitor agent sessions and performance

### 2. **Team Management**
- Create and manage teams of agents
- Define team hierarchies and relationships
- Configure team-specific workflows
- Track team performance metrics

### 3. **Tools Integration**
- Integrate external tools and services
- Configure tool-specific parameters
- Monitor tool usage and performance

### 4. **Security**
- Clerk-based authentication
- Session management
- PHI-compliant data handling
- Secure knowledge base access

## Contributing

We welcome contributions to InfiniteRegen! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Open a pull request

## Contact

For questions or feedback, please reach out to [hoomandigital18@gmail.com](mailto:hoomandigital18@gmail.com)
