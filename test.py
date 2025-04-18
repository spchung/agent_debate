# beautify log
from src.debate.workers import beautifier_agent

filename = f"debate_results/Planned_Against_AI_Regulation_vs_Basic_For_AI_Regulation.txt"
raw_text = open(filename, "r").read()

ff = open(f"debate_results/test.md", "w+")

res = beautifier_agent.run(
    beautifier_agent.input_schema(
        debate_log_raw_text=raw_text
    )
)
ff.write(res.debate_log_md)
ff.close()