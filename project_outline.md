# Debate Agent Project Plan

## Project Overview
An investigation into how different planning strategies affect debate performance between autonomous agents in a language-based environment.

## Environment
- **Type:** Language environment where two debate agents argue for/against a proposed topic
- **Resources:** Agents may access documents (research papers, news articles, etc.) through a retrieval workflow

## Autonomous Agents
Five distinct agent types with different planning strategies:

1. **Naive Agents**
   - No knowledge base integration
   - Responses tuned only by initial prompt
   - Reacts to immediate previous arguments

2. **Knowledge Base Agents**
   - Access to a set of reference documents
   - Consults knowledge base before generating responses
   - Focuses on fact retrieval and evidence-based arguments

3. **Planning-Based Agents**
   - Reads knowledge base before first round
   - Pre-generates points and counter-arguments for anticipated topics
   - When opponent addresses a predicted point, deploys pre-generated counter-arguments

4. **Knowledge Graph Agents**
   - Builds and maintains a knowledge graph as the debate progresses
   - Maps conceptual relationships between arguments and evidence
   - Can maintain and expand graph across multiple sequential debates

5. **Strategic State-Based Agents (Not Implemented)**
   - Dynamically switches between tactical states based on opponent's argument strength:
     - *Offensive:* Attacking opponent positions
     - *Defensive:* Strengthening own positions
     - *Exploratory:* Introducing new arguments
     - *Consolidating:* Reinforcing strongest points

## Research Question
**How do different planning strategies affect the persuasiveness and coherence of arguments in multi-turn debates?**

## Experiment Design
- **Format:** Round-robin tournament
  - Each agent type debates against all others in head-to-head matches
  - Fixed number of debate rounds per match
  - Complete debate transcripts generated for evaluation

- **Topics:** Multiple debate topics to test strategy effectiveness across domains

## Evaluation Methodology

### Automated Evaluation
- **Argument Coherence:** NLP metrics for textual entailment and logical flow
- **Topic Adherence:** Topic modeling to measure relevance and focus
- **LLM Judge:** Reasoning-optimized LLM model as impartial evaluator

### Human Evaluation
- **Blind Evaluation:** Transcripts presented with agent identities hidden
- **Rubric-Based Scoring:** Detailed criteria for evaluators to follow via survey
- **Win/Loss Determination:** Overall assessment of which agent performed better

## Analysis Approach
- Identify which planning strategies perform best overall
- Determine if strategy effectiveness varies by topic domain
- Compare automated metrics against human judgments
- Evaluate consistency of performance across different opponents