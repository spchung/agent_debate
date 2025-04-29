import os
from src.evaluation.evaluation import evaluate

'''
Anon transcript to markdown mapping
'''

def eval_all():
    # evaluate the debate log file
    topic = "Self-regulation by the AI industry is preferable to government regulation."
    
    for file in os.listdir('debate_results'):
        skip = False
        # check if already evaluated
        filename = file.split('.')[0]
        for result_json_file in os.listdir('evaluation_results'):
            if filename in result_json_file:
                print (f"SKIP: Already evaluated {file}...")
                skip = True
                break
        
        if file.endswith('.md') and not skip:
            file_path = os.path.join('debate_results', file)
            destination_dir = 'evaluation_results'
            print (f"Evaluating {file_path}...")
            evaluate(file_path, topic, destination_dir)

def eval_one(file_path):
    # evaluate the debate log file
    topic = "Self-regulation by the AI industry is preferable to government regulation."
    
    destination_dir = 'evaluation_results'
    print (f"Evaluating {file_path}...")
    evaluate(file_path, topic, destination_dir)

if __name__ == "__main__":
    # eval_one('debate_results/kb_for_vs_basic_against_0.md')
    eval_all()