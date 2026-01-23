# Hierarchical Multiagent LangGraph
![ROS2](https://img.shields.io/badge/ros2-jazzy-blue?logo=ros&logoColor=white)
![License](https://img.shields.io/github/license/grupo-avispa/hierarchical_multiagent_langgraph)
[![Build](https://github.com/grupo-avispa/hierarchical_multiagent_langgraph/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/grupo-avispa/hierarchical_multiagent_langgraph/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/grupo-avispa/hierarchical_multiagent_langgraph/graph/badge.svg?token=R48HZO62SQ)](https://codecov.io/gh/grupo-avispa/hierarchical_multiagent_langgraph)

This package implements a hierarchical multi-agent system where a **Supervisor** coordinates multiple **Single-Purpose Agents (SPAs)** to execute complex tasks. The system uses LLM models (via Ollama) for reasoning and decision-making, and LangGraph to orchestrate state graph-based workflows. Designed for native ROS2 integration, it enables robots to decompose user queries into subtasks and execute them in parallel through specialized agents.

## Key Features

- **Hierarchical architecture**: Supervisor → Specialized agents
- **LLM integration**: Ollama-based reasoning for intelligent decisions
- **Concurrent execution**: Multiple agents running in independent threads
- **Model Context Protocol (MCP)**: Support for external tools via MCP servers
- **Lifecycle management**: Dynamic agent creation, monitoring, and deletion
- **ROS2 integration**: Native service for receiving user queries

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS) 2](https://docs.ros.org/en/jazzy/) (middleware for robotics),
- [langgraph_base_ros](https://github.com/grupo-avispa/langgraph_base_ros) (base for LangGraph-ROS2 integration),
- [llm_interactions_msgs](https://github.com/grupo-avispa/llm_interactions_msgs) (messages for LLM interactions),
- [langchain](https://github.com/langchain-ai/langchain) (framework for developing LLM applications)
- [langgraph](https://github.com/langchain-ai/langgraph) (framework for building language-based state graphs)

#### Building

To build from source, clone the latest version from the main repository into your colcon workspace and compile the package using

```bash
cd colcon_workspace/src
git clone https://github.com/grupo-avispa/hierarchical_multiagent_langgraph.git -b jazzy
cd ../
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build --symlink-install
```

## Usage

With some scan source running, run the hierarchical_multiagent_langgraph node with:

```bash
ros2 launch hierarchical_multiagent_langgraph langgraph.launch.py
```

Then, the agent can be called with:

```bash
ros2 service call /hierarchical_multiagent/call_agent llm_interactions_msgs/srv/CallAgent "{query: 'Prepare a cup of coffee'}"
```

## System Architecture

### Overview

```mermaid
graph TB
    subgraph ROS2["ROS2 Environment"]
        SRV[/"CallAgent Service"/]
        TMR["Agent Timer<br/>(1s period)"]
    end

    subgraph HMA["HierarchicalMultiagent Node"]
        SM["SupervisorManager"]
        LOOP["Main Event Loop"]
    end

    subgraph AgentManagement["Agent Lifecycle"]
        PAL["Pending Agents List"]
        RAL["Running Agents List"]
        FAL["Finished Agents List"]
    end

    subgraph Agents["Single-Purpose Agents"]
        SPA1["Agent #1<br/>Thread + Event Loop"]
        SPA2["Agent #2<br/>Thread + Event Loop"]
        SPA3["Agent #N<br/>Thread + Event Loop"]
    end

    subgraph LLM["LLM Backend"]
        OLLAMA["Ollama Server"]
        MCP["MCP Servers<br/>(Optional)"]
    end

    SRV -->|"User Query"| SM
    SM -->|"create_agent"| PAL
    TMR -->|"Consume"| PAL
    PAL -->|"Execute"| RAL
    RAL --> SPA1
    RAL --> SPA2
    RAL --> SPA3
    SPA1 -->|"Complete"| FAL
    SPA2 -->|"Complete"| FAL
    SPA3 -->|"Complete"| FAL
    SM <-->|"Reasoning"| OLLAMA
    SPA1 <-->|"Reasoning"| OLLAMA
    SPA2 <-->|"Reasoning"| OLLAMA
    SPA1 <-.->|"Tools"| MCP
```

### Supervisor Workflow

The Supervisor uses a LangGraph state graph to analyze tasks and manage agents:

```mermaid
stateDiagram-v2
    [*] --> set_initial_messages: START
    set_initial_messages --> analyze_task: Render system prompt
    
    analyze_task --> analyze_task: route='agent'<br/>(No tool call, retry)
    analyze_task --> finalize_conversation: route='finish'<br/>(Tool call detected<br/>OR max_steps)
    
    finalize_conversation --> [*]: END

    note right of analyze_task
        LLM decides:
        - create_agent(query)
        - delete_agent(id)
        - skip_agent()
    end note
```

### Agent Workflow (SPA)

Each Single-Purpose Agent executes its own LangGraph graph to complete tasks:

```mermaid
stateDiagram-v2
    [*] --> query_response: START
    
    query_response --> query_response: route='agent'<br/>(Tool call & steps < max)
    query_response --> finish_ollama_interaction: route='finish'<br/>(No tool call<br/>OR max_steps)
    
    finish_ollama_interaction --> [*]: END
    
    note right of query_response
        LLM reasoning loop:
        - Process messages
        - Call tools (MCP/local)
        - Generate response
    end note
```

### Agent Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant Service as CallAgent Service
    participant Sup as SupervisorManager
    participant Timer as Agent Timer
    participant SPA as SinglePurposeAgent
    participant LLM as Ollama

    User->>Service: query="Prepare coffee"
    Service->>Sup: InputState{user_prompt}
    
    Sup->>LLM: analyze_task()
    LLM-->>Sup: create_agent("Prepare coffee")
    
    Sup->>Sup: Add to pending_agents_list
    
    loop Every 1 second
        Timer->>Sup: Check pending_agents_list
        Sup-->>Timer: AgentTask
        Timer->>SPA: Create new event loop
        Timer->>SPA: run_agent(initial_state)
        
        loop Agent reasoning
            SPA->>LLM: query_response()
            LLM-->>SPA: Tool call / Response
            SPA->>SPA: Execute tool
        end
        
        SPA-->>Sup: FinishedAgentsState
    end
    
    Sup->>Sup: Move to finished_agents_list
    Service-->>User: "Query submitted"
```

### Supervisor Tools

```mermaid
graph LR
    subgraph SupervisorTools["Supervisor Tools"]
        CT["create_agent(query)"]
        DT["delete_agent(agent_id)"]
        ST["skip_agent()"]
    end

    subgraph Effects["Effects"]
        PAL["→ pending_agents_list"]
        RAL["→ running_agents_list<br/>(cancel + remove)"]
        NOOP["→ No action"]
    end

    CT --> PAL
    DT --> RAL
    ST --> NOOP
```

### Agent States

```mermaid
stateDiagram-v2
    [*] --> IDLE: Agent created
    IDLE --> RUNNING: Timer consumes
    RUNNING --> SUCCESS: Task completed<br/>(steps < max_steps)
    RUNNING --> FAILURE: Error / Cancelled<br/>/ max_steps reached
    SUCCESS --> [*]
    FAILURE --> [*]
```

## Main Components

### HierarchicalMultiagent (main.py)

ROS2 node that manages the system lifecycle:

- Exposes `CallAgent` service to receive queries
- Runs periodic timer to consume pending agents
- Initializes and configures the `SupervisorManager`
- Manages multiple execution threads for agents

### SupervisorManager (supervisor.py)

Orchestrates multi-agent coordination:

- **LLM Tools**: `create_agent`, `delete_agent`, `skip_agent`
- **State management**: Thread-safe lists for pending/running/finished agents
- **LangGraph graph**: Analysis and decision workflow
- **Mutex protection**: `agent_lists_lock` for safe concurrent access

### SinglePurposeAgent (agent.py)

Specialized agent for individual tasks:

- **States**: `IDLE`, `RUNNING`, `SUCCESS`, `FAILURE`
- **MCP integration**: Access to external tools
- **Own graph**: Independent reasoning cycle
- **Isolation**: Separate event loop per agent


