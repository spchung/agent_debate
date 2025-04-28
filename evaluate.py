from src.evaluation.evaluation import evaluate

if __name__ == "__main__":
    # evaluate the debate log file
    topic = "Self-regulation by the AI industry is preferable to government regulation."
    evaluate("example_debate_transcript.md", topic)
