## structure

src/
├── agents/
│   ├── base_agent.py            # Abstract base class for all agents
│   ├── advocate_agent.py        # Implementation of the advocate agent
│   ├── critic_agent.py          # Implementation of the critic agent
│   ├── synthesis_agent.py       # Optional synthesis/moderator agent
│   └── utils/
│       ├── memory.py            # Different memory implementations
│       └── reasoning.py         # Reasoning strategy implementations
├── debate/
│   ├── debate_manager.py        # Orchestrates the debate process
│   ├── debate_formats.py        # Different debate structures/protocols
│   ├── evaluation.py            # Metrics for evaluating debate quality
│   └── visualization.py         # Tools for visualizing debate results
├── paper_processing/
│   ├── parser.py                # Extract content and structure from papers
│   ├── reference_manager.py     # Track and retrieve paper references
│   └── data_loader.py           # Handle loading different paper formats
├── experiments/
│   ├── experiment_runner.py     # Configure and run sets of experiments
│   ├── configurations/          # Different experimental configurations
│   └── results/                 # Store experimental results
├── ui/                          # Optional: Simple interface for viewing debates
└── main.py                      # Entry point for running the system