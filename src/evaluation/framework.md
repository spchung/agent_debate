# Simplified Debate Agent Evaluation Framework

## Core Evaluation Categories (100 points total)

### 0. Opening and Closing Statements (20 points)
| Criterion | Description | Points |
|-----------|-------------|--------|
| **Opening Effectiveness** | Clear framing of position; introduces key arguments | 0-10 |
| **Closing Effectiveness** | Effectively summarizes key points; reinforces position | 0-10 |

### 1. Argument Quality (30 points)
| Criterion | Description | Points |
|-----------|-------------|--------|
| **Evidence Strength** | How well the agent uses available resources to support claims | 0-15 |
| **Logical Reasoning** | Quality of logical connections between premises and conclusions | 0-15 |

### 2. Coherence (25 points)
| Criterion | Description | Points |
|-----------|-------------|--------|
| **Stance Consistency** | Maintaining consistent position throughout the debate | 0-15 |
| **Argument Development** | Building upon previous points rather than abandoning them | 0-10 |

### 3. Rebuttal Effectiveness (25 points)
| Criterion | Description | Points |
|-----------|-------------|--------|
| **Counter-Argument Quality** | Directly challenging opponent's key premises and conclusions | 0-15 |
| **Adaptive Response** | Adjusting strategy based on opponent's arguments | 0-10 |

## Scoring Guidelines

### Opening Effectiveness (0-10)
- **8-10:** Exceptionally clear framing; introduces 2-3 strong key arguments that set the foundation for the debate; establishes a compelling stance
- **6-7:** Good framing with clear arguments; establishes position effectively but may lack some persuasive impact
- **3-5:** Basic introduction of position but lacks clear argument structure or compelling framing
- **0-2:** Unclear position; fails to establish key arguments or provides weak foundation for later rounds

### Closing Effectiveness (0-10)
- **8-10:** Masterfully synthesizes all key points; directly addresses main counterarguments; leaves lasting impression reinforcing position
- **6-7:** Good summary of main points; generally ties arguments together but may miss some opportunities to reinforce position
- **3-5:** Basic attempt at summarizing but fails to effectively tie different arguments together; limited reinforcement of position
- **0-2:** Weak conclusion that fails to summarize debate points; introduces new arguments or abandons earlier points without synthesis

### Evidence Strength (0-15)
- **12-15:** Consistently cites specific, relevant evidence from resources; uses evidence appropriately to support claims
- **9-11:** Generally uses evidence well but occasionally overreaches or misapplies
- **6-8:** Uses some evidence but often makes unsupported claims
- **3-5:** Minimal evidence usage; primarily relies on assertions
- **0-2:** Little to no evidence used from available resources

### Logical Reasoning (0-15)
- **12-15:** Arguments follow clear logical structure; conclusions follow directly from premises
- **9-11:** Generally logical but contains occasional gaps or weak connections
- **6-8:** Some logical structure but contains several fallacies or non-sequiturs
- **3-5:** Frequent logical errors; conclusions often don't follow from premises
- **0-2:** Little discernible logical structure to arguments

### Stance Consistency (0-15)
- **12-15:** Maintains perfectly consistent position; no contradictions
- **8-11:** Generally consistent with minor shifts in emphasis
- **4-7:** Some contradictions or significant position shifts during debate
- **0-3:** Frequently contradicts previous statements or abandons initial position

### Argument Development (0-10)
- **8-10:** Systematically builds on previous points; creates cumulative case
- **6-7:** Generally builds on previous arguments with occasional disconnects
- **3-5:** Limited connection between arguments across rounds
- **0-2:** Each round presents essentially new arguments with little continuity

### Counter-Argument Quality (0-15)
- **12-15:** Directly addresses opponent's strongest points with effective counters
- **8-11:** Addresses most opponent points but misses some key arguments
- **4-7:** Addresses some opponent points but often focuses on weaker arguments
- **0-3:** Rarely engages with opponent's actual arguments

### Adaptive Response (0-10)
- **8-10:** Skillfully adjusts strategy based on opponent's approach
- **6-7:** Shows some adaptation to opponent's arguments
- **3-5:** Minimal adaptation; mostly follows pre-planned approach
- **0-2:** No discernible adaptation to opponent's arguments or strategy

## Sample Evaluation Prompt for LLM

```
Evaluate the debate between two agents on [TOPIC]. Focus on these four categories:

0. Opening and Closing Statements (20 points):
   - Opening Effectiveness (0-10): How well does the agent frame their position and introduce key arguments?
   - Closing Effectiveness (0-10): How effectively does the agent summarize key points and reinforce their position?

1. Argument Quality (30 points):
   - Evidence Strength (0-15): How well does the agent use available resources?
   - Logical Reasoning (0-15): How sound is the agent's logical structure?

2. Coherence (25 points):
   - Stance Consistency (0-15): Does the agent maintain a consistent position?
   - Argument Development (0-10): Does the agent build upon their previous points?

3. Rebuttal Effectiveness (25 points):
   - Counter-Argument Quality (0-15): How well does the agent address opponent's key points?
   - Adaptive Response (0-10): Does the agent adjust strategy based on the opponent?

For each criterion, provide:
- A specific score
- 1-2 sentences explaining your reasoning
- An example from the debate illustrating your assessment

Available resources: [RESOURCES]
Debate transcript: [TRANSCRIPT]
```