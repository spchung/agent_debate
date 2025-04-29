import re, json
from typing import List, Tuple
from src.evaluation.workers.opening_statement import opening_statement_judge
from src.evaluation.workers.closing_statement import closing_statement_judge
from src.evaluation.workers.argument_quality import argument_quality_judge
from src.evaluation.workers.rebuttal_quality import rebuttal_quality_judge, StatementAndRebuttalPairModel
from src.evaluation.workers.coherence_quality import coherence_quality_judge

def evaluate_opening(statement: str, topic: str):
    scoring = opening_statement_judge.run(
        opening_statement_judge.input_schema(
            opening_statement=statement,
            topic=topic,
        )
    )
    
    res = scoring.model_dump()
    res['category'] = "Opening Statement"
    return res

def evaluate_closing(statement: str, topic: str):
    scoring = closing_statement_judge.run(
        closing_statement_judge.input_schema(
            closing_statement=statement,
            topic=topic,
        )
    )

    res = scoring.model_dump()
    res['category'] = "Closing Statement"
    return res

def evaluate_argument_quality(arguments: List[str], topic: str):
    scoring = argument_quality_judge.run(
        argument_quality_judge.input_schema(
            topic=topic,
            arguments=arguments,
        )
    )

    res = scoring.model_dump()
    res['category'] = "Argument Quality"
    return res

def evaluate_rebuttal_quality(
    statement_and_rebuttal_pairs: List[StatementAndRebuttalPairModel], 
    topic: str
):
    scoring = rebuttal_quality_judge.run(
        rebuttal_quality_judge.input_schema(
            topic=topic,
            statement_and_rebuttal_pairs=statement_and_rebuttal_pairs,
        )
    )
    
    res = scoring.model_dump()
    res['category'] = "Rebuttal Quality"
    return res

def evaluate_coherence_quality(arguments: List[str], topic: str):
    scoring = coherence_quality_judge.run(
        coherence_quality_judge.input_schema(
            topic=topic,
            arguments=arguments,
        )
    )
    
    res = scoring.model_dump()
    res['category'] = "Coherence Quality"
    return res

def evaluate(
        file_path,
        topic,
        destination_dir,
    ):
    with open(file_path, 'r') as file:
        text = file.read()

    # total scoring objects
    for_agent_total_scoring = [] # (scoring category, score + reasoning model)
    against_agent_total_scoring = []
    
    # for agent 
    for_agent_re = r"\*\*\[For_Agent\]:\*\*\s*([\s\S]*?)(?=\n\n\*\*\[Against_Agent\]:|$|\n\n###)"
    for_agent_matches = re.findall(for_agent_re, text)

    # against agent
    against_agent_re = r"\*\*\[Against_Agent\]:\*\*\s*([\s\S]*?)(?=\n\n\*\*\[For_Agent\]:|$|\n\n###)"
    against_agent_matches = re.findall(against_agent_re, text)
    
    assert len(for_agent_matches) == len(against_agent_matches), "Mismatch in number of rounds between agents."

    rounds = [] # [(FOR),(AGAINST)]
    for i in range(len(for_agent_matches)):
        tup = (for_agent_matches[i], against_agent_matches[i])
        rounds.append(tup)

    # 1. judge opening statement
    for_opening_statement, against_opening_statement = rounds[0]
    
    for_opening_statement_score = evaluate_opening(for_opening_statement, topic)
    against_opening_statement_score = evaluate_opening(against_opening_statement, topic)

    # scoring
    # for_agent_total_scoring.append(("Opening Statement", for_opening_statement_score.model_dump()))
    for_agent_total_scoring.append(for_opening_statement_score)
    against_agent_total_scoring.append(against_opening_statement_score)
    

    # 2. judge argument quality 
    for_arguments = []
    against_arguments = []
    for i in range(len(rounds)):
        for_arguments.append(rounds[i][0])
        against_arguments.append(rounds[i][1])

    for_agent_argument_score = evaluate_argument_quality(for_arguments, topic)
    against_agent_argument_score = evaluate_argument_quality(against_arguments, topic)

    # scoring
    for_agent_total_scoring.append(for_agent_argument_score)
    against_agent_total_scoring.append(against_agent_argument_score)

    # 3. judge coherence
    for_agent_coherence_score = evaluate_coherence_quality(for_arguments, topic)
    against_agent_coherence_score = evaluate_coherence_quality(against_arguments, topic)

    # scoring
    for_agent_total_scoring.append(for_agent_coherence_score)
    against_agent_total_scoring.append(against_agent_coherence_score)

    # 4. judge rebuttal quality
    for_agent_rebuttal = [] #(opp_arg, rebuttal)
    against_agent_rebuttal = []

    for i in range(1, len(rounds)-1):
        prev_for_msg, prev_against_msg = rounds[i-1]
        for_msg, against_msg = rounds[i]

        for_agent_rebuttal.append(
            StatementAndRebuttalPairModel(
                statement=prev_against_msg,
                rebuttal=for_msg
            )
        )
        
        against_agent_rebuttal.append(
            StatementAndRebuttalPairModel(
                statement=prev_for_msg,
                rebuttal=against_msg
            )
        )
    
    for_agent_rebuttal_score = evaluate_rebuttal_quality(for_agent_rebuttal, topic)
    against_agent_rebuttal_score = evaluate_rebuttal_quality(against_agent_rebuttal, topic)

    # scoring
    for_agent_total_scoring.append(for_agent_rebuttal_score)
    against_agent_total_scoring.append(against_agent_rebuttal_score)

    # 5. judge closing statement
    for_closing_statement, against_closing_statement = rounds[-1]
    
    for_agent_previous_statements = []
    against_agent_previous_statement = []
    
    for i in range(len(rounds)-1):
        for_agent_previous_statements.append(rounds[i][0])
        against_agent_previous_statement.append(rounds[i][-1])
    
    for_closing_statement_score = evaluate_closing(for_closing_statement, topic)
    against_closing_statement_score = evaluate_closing(against_closing_statement, topic)
    
    # scoring
    for_agent_total_scoring.append(for_closing_statement_score)
    against_agent_total_scoring.append(against_closing_statement_score)

    # aggregate scores
    for_agent_total_score = 0
    for section in for_agent_total_scoring:
        for_agent_total_score += section['score']
    
    against_agent_total_score = 0
    for section in against_agent_total_scoring:
        against_agent_total_score += section['score']
    
    d = {
        "for_agent": for_agent_total_scoring,
        "for_total_score": for_agent_total_score,
        "against_agent": against_agent_total_scoring,
        "against_total_score": against_agent_total_score,
    }

    filename = file_path.split("/")[-1]
    filename = filename.split(".")[0]

    with open(f"{destination_dir}/{filename}_evaluation_results.json", "w") as f:
        json.dump(d, f, indent=4)